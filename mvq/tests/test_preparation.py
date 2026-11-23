"""Preparation contracts using real Lhotse serialization and the package codec."""
import gzip
import json
import sqlite3
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from lhotse import CutSet, Recording
from lhotse.cut import MonoCut
from lhotse.features.io import NumpyHdf5Writer
from lhotse.utils import fastcopy
from multi_quantization import Quantizer

from mvq.common import (EMBEDDING_DIM, FRAME_SHIFT, absolute_audio_cut, check_array,
                        encode_frames, load_quantizer, sha256)
from mvq.extract import batches, merge_outputs, partition_inputs
from mvq.train_quantizer import channel_stats, frame_batches


class Preparation(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.audio = self.root / 'audio.wav'
        with wave.open(str(self.audio), 'wb') as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(np.zeros(32000, dtype=np.int16).tobytes())
        self.recording = Recording.from_file(self.audio)

    def tearDown(self):
        self.tmp.cleanup()

    def cut(self, name='cut', duration=1, start=0):
        return MonoCut(id=name, start=start, duration=duration, channel=0, recording=self.recording)

    def store(self, cut, values, field='codebook_indexes'):
        with NumpyHdf5Writer(self.root / f'arrays-{cut.id}') as writer:
            array = writer.store_array(key='key', value=values, temporal_dim=0,
                                       frame_shift=FRAME_SHIFT, start=cut.start)
        return fastcopy(cut, custom={field: array})

    def test_batches_keep_partial_tails_and_respect_padding(self):
        cuts = [SimpleNamespace(id=str(i), duration=d) for i, d in enumerate([15.43, 15.31, 3.08, 14.16, 15.69])]
        result = list(batches(cuts, max_duration=32, max_cuts=2))
        self.assertEqual([c.id for batch in result for c in batch], [c.id for c in cuts])
        self.assertTrue(all(max(c.duration for c in b) * len(b) <= 32 for b in result))
        with self.assertRaisesRegex(ValueError, 'single cut'):
            list(batches(cuts, max_duration=3, max_cuts=2))

    def test_unsigned_hdf5_roundtrip_and_nonzero_start(self):
        values = np.full((50, 8), 1001, dtype=np.uint16)
        cut = self.store(self.cut(start=0.25), values)
        manifest = self.root / 'cuts.jsonl.gz'
        CutSet.from_cuts([cut]).to_file(manifest)
        loaded = next(iter(CutSet.from_file(manifest)))
        array = check_array(loaded, 'codebook_indexes', 8, integer=True, codebook_size=1024)
        np.testing.assert_array_equal(array, values)
        self.assertEqual(array.dtype, np.uint16)
        self.assertEqual(loaded.codebook_indexes.start, 0.25)
        with self.assertRaisesRegex(ValueError, 'out-of-range'):
            check_array(loaded, 'codebook_indexes', 8, integer=True, codebook_size=256)

    def test_partition_and_merge_preserve_input_order(self):
        cuts = [self.cut(f'cut-{i}', start=0.1) for i in range(5)]
        source = self.root / 'source.jsonl.gz'
        CutSet.from_cuts(cuts).to_file(source)
        args = SimpleNamespace(input_cuts=source, num_gpus=2, num_workers=2, audio_root=self.root)
        with sqlite3.connect(':memory:') as connection:
            paths, count = partition_inputs(args, self.root, connection)
            self.assertEqual(count, 5)
            self.assertEqual(sorted(c.id for rank in paths for p in rank for c in CutSet.from_file(p)),
                             sorted(c.id for c in cuts))
            outputs = []
            for i, cut in enumerate(reversed(cuts)):
                out = self.root / f'out-{i}.jsonl.gz'
                CutSet.from_cuts([self.store(cut, np.full((50, 8), i, dtype=np.uint8))]).to_file(out)
                outputs.append(out)
            final = self.root / 'joined.jsonl.gz'
            self.assertEqual(merge_outputs(connection, outputs, final, 'codebook_indexes', 8), 5)
            self.assertEqual([c.id for c in CutSet.from_file(final)], [c.id for c in cuts])

    def test_duplicate_input_rejected(self):
        source = self.root / 'duplicates.jsonl.gz'
        with gzip.open(source, 'wt') as stream:
            payload = json.dumps(self.cut().to_dict()) + '\n'
            stream.write(payload * 2)
        args = SimpleNamespace(input_cuts=source, num_gpus=1, num_workers=0, audio_root=self.root)
        with sqlite3.connect(':memory:') as connection, self.assertRaisesRegex(ValueError, 'Duplicate input'):
            partition_inputs(args, self.root, connection)

    def test_merge_rejects_missing_or_wrong_interval(self):
        source = self.root / 'source.jsonl.gz'
        CutSet.from_cuts([self.cut()]).to_file(source)
        args = SimpleNamespace(input_cuts=source, num_gpus=1, num_workers=0, audio_root=self.root)
        with sqlite3.connect(':memory:') as connection:
            partition_inputs(args, self.root, connection)
            with self.assertRaisesRegex(ValueError, 'coverage mismatch'):
                merge_outputs(connection, [], self.root / 'missing.jsonl.gz', 'codebook_indexes', 8)
        wrong = self.root / 'wrong.jsonl.gz'
        CutSet.from_cuts([self.store(self.cut(start=0.2), np.zeros((50, 8), dtype=np.uint8))]).to_file(wrong)
        with sqlite3.connect(':memory:') as connection:
            partition_inputs(args, self.root, connection)
            with self.assertRaisesRegex(ValueError, 'input interval'):
                merge_outputs(connection, [wrong], self.root / 'mismatch.jsonl.gz', 'codebook_indexes', 8)

    def test_frame_stream_and_normalization_statistics(self):
        arrays, cuts = [], []
        for i, size in enumerate([11, 7, 4]):
            array = np.arange(sum(len(a) for a in arrays) * 3,
                              (sum(len(a) for a in arrays) + size) * 3, dtype=np.float32).reshape(size, 3)
            arrays.append(array)
            cuts.append(self.store(self.cut(f'emb-{i}', size * FRAME_SHIFT), array, 'embedding'))
        manifest = self.root / 'embeddings.jsonl.gz'
        CutSet.from_cuts(cuts).to_file(manifest)
        streamed = list(frame_batches(manifest, 'embedding', 3, 8, 4, 42, repeat=False))
        expected = np.concatenate(arrays)
        actual = np.concatenate(streamed)
        np.testing.assert_array_equal(np.sort(actual[:, 0]), np.sort(expected[:, 0]))
        self.assertTrue(all(len(b) <= 4 for b in streamed))
        mean, std = channel_stats(manifest, 'embedding', 3)
        np.testing.assert_allclose(mean.numpy(), expected.mean(axis=0))
        np.testing.assert_allclose(std.numpy(), expected.std(axis=0), rtol=1e-6)

    def test_codec_matches_package_for_256_and_1024(self):
        for size in (256, 1024):
            torch.manual_seed(1)
            quantizer = Quantizer(dim=EMBEDDING_DIM, num_codebooks=8, codebook_size=size).eval()
            frames = torch.randn(5, EMBEDDING_DIM)
            with torch.no_grad():
                expected = quantizer.encode(frames, as_bytes=False).numpy()
            actual = encode_frames(quantizer, frames, size, 2)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(actual.dtype, np.uint8 if size == 256 else np.uint16)
            if size == 1024:
                self.assertGreater(int(actual.max()), 255)

    def test_checkpoint_normalization_and_teacher_mismatch(self):
        quantizer = Quantizer(dim=EMBEDDING_DIM, num_codebooks=8, codebook_size=256)
        checkpoint = self.root / 'quantizer.pt'
        identity = dict(config_sha256='config', weights_sha256={'weights': 'a'},
                        embedding_layer=-1, embedding_dim=EMBEDDING_DIM, frame_shift=FRAME_SHIFT)
        mean, std = torch.ones(EMBEDDING_DIM), torch.full((EMBEDDING_DIM,), 2.0)
        saved = dict(quantizer=quantizer.state_dict(), mean=mean, std=std, metadata={'teacher': identity})
        torch.save(saved, checkpoint)
        loaded, loaded_mean, loaded_std = load_quantizer(checkpoint, 8, 256, 'cpu', teacher=identity)
        frames = torch.randn(3, EMBEDDING_DIM)
        with torch.no_grad():
            expected = quantizer.encode((frames - mean) / std, as_bytes=False).numpy()
        np.testing.assert_array_equal(encode_frames(loaded, frames, 256, 2, loaded_mean, loaded_std), expected)
        with self.assertRaisesRegex(ValueError, 'normalization disagrees'):
            load_quantizer(checkpoint, 8, 256, 'cpu', 'none')
        with self.assertRaisesRegex(ValueError, 'different teacher'):
            load_quantizer(checkpoint, 8, 256, 'cpu', teacher={**identity, 'embedding_layer': 0})
        saved['std'][0] = 0
        torch.save(saved, checkpoint)
        with self.assertRaisesRegex(ValueError, 'positive std'):
            load_quantizer(checkpoint, 8, 256, 'cpu')
        torch.save(quantizer.state_dict(), checkpoint)
        _, mean, std = load_quantizer(checkpoint, 8, 256, 'cpu', 'none')
        self.assertIsNone(mean)
        self.assertIsNone(std)

    def test_absolute_audio_and_missing_files(self):
        cut = self.cut()
        original = cut.recording.sources[0].source
        cut.recording.sources[0].source = 'audio.wav'
        prepared = absolute_audio_cut(cut, self.root)
        self.assertEqual(prepared.recording.sources[0].source, original)
        self.assertEqual(cut.recording.sources[0].source, 'audio.wav')
        with self.assertRaises(FileNotFoundError):
            absolute_audio_cut(cut, self.root / 'missing')

    def test_source_hashes(self):
        project = Path(__file__).resolve().parents[2]
        for name in ('zipformer2.py', 'scaling.py', 'subsampling.py', 'encoder_interface.py', 'optim.py'):
            expected = json.loads((project / 'SOURCE_SHA256.json').read_text())[name]
            self.assertEqual(sha256(project / name), expected)
        for name, expected in json.loads((project / 'mvq/SOURCE_SHA256.json').read_text()).items():
            self.assertEqual(sha256(project / 'mvq/spear_hf' / name), expected)


if __name__ == '__main__':
    unittest.main()
