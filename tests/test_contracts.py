import copy
import importlib
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from torch import nn
from lhotse import CutSet
from config import parse_args, teachers
from data import LibriSpeechDataset, load_targets, mix_targets, validation_cut, validate_cut, validation_rank
from model import MVQModel, align_targets, build_model
from masking import compute_mask_indices, compute_mask_indices_block
from utils import (seed_everything, rng_state, restore_rng, check_resume, config_dict,
                   module_runtime_state, restore_module_runtime)
from local.make_smoke_manifest import create_manifest


def args(extra=()):
    return parse_args(["--train-cuts", "unused.jsonl.gz", "--teacher-routing", "both", *extra])


def small_args(extra=()):
    return args(["--num-encoder-layers", "1,1,1", "--downsampling-factor", "1,2,1",
                 "--encoder-dim", "64", "--encoder-unmasked-dim", "32", "--feedforward-dim", "128",
                 "--num-heads", "2", "--cnn-module-kernel", "7", "--query-head-dim", "16",
                 "--value-head-dim", "8", "--pos-head-dim", "4", *extra])


class IdentityEmbed(nn.Module):
    def forward(self, x, lens):
        return x, lens


class IdentityEncoder(nn.Module):
    def forward(self, x, lens, padding):
        return x, lens


def reference_class():
    path = Path(__file__).resolve().parents[2] / "unified_encoder" / "zipformer_audio_encoder"
    if not path.exists():
        raise unittest.SkipTest("Optional original source checkout is unavailable")
    sys.path.insert(0, str(path))
    try:
        return importlib.import_module("model_multi_kd_multi_teacher").MultiKDModel
    finally:
        sys.path.remove(str(path))


class Contracts(unittest.TestCase):
    def test_metadata_and_routing(self):
        original = parse_args(["--train-cuts", "x"])
        t = teachers(original)
        self.assertEqual([(s.num_codebooks, s.frame_rate, s.repeat, s.active) for s in t],
                         [(16, 50, 1, True), (8, 25, 2, False)])
        self.assertTrue(all(t.active for t in teachers(args())))
        self.assertEqual([t.active for t in teachers(args(["--teacher-routing", "teacher2"]))], [False, True])

    def test_config_and_overrides(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text(json.dumps(dict(teacher_routing="both", mask_prob=0.5, output_downsampling_factor=1)))
            a = parse_args(["--config", str(p), "--train-cuts", "x", "--mask-prob", "0.25"])
            self.assertEqual(a.mask_prob, 0.25)
            self.assertTrue(teachers(a)[1].active)

    def test_invalid_metadata(self):
        with self.assertRaises(ValueError):
            args(["--teacher-frame-rate", "50,30"])
        with self.assertRaises(ValueError):
            args(["--distillation-delta", "0,-1"])
        with self.assertRaises(ValueError):
            args(["--num-codebooks", "16"])

    def test_per_utterance_alignment(self):
        a = torch.tensor([[[4], [8], [-100]], [[9], [7], [1]]])
        result = align_targets(a, torch.tensor([2, 3]), torch.tensor([3, 5]), 5, 2)
        self.assertEqual(result[:, :, 0].tolist(), [[4, 4, 8, -100, -100], [9, 9, 7, 7, 1]])
        with self.assertRaises(ValueError):
            align_targets(a, torch.tensor([2, 3]), torch.tensor([30, 30]), 30, 2)

    def test_delta_and_reference_loss_gradients(self):
        a = args(["--num-codebooks", "2,2", "--codebook-size", "256,256", "--distillation-delta", "3,5"])
        specs = teachers(a)
        new = MVQModel(IdentityEmbed(), IdentityEncoder(), 128, specs, a).eval()
        old = reference_class()(IdentityEmbed(), IdentityEncoder(), encoder_dim=128,
            num_codebooks=(2, 2), teacher_frame_ratio=(1, 1), distillation_layer=(-1, -1),
            distillation_delta=(3, 5), n_mels=128).eval()
        old.codebook_loss_heads.load_state_dict(new.codebook_loss_heads.state_dict())
        x = torch.randn(2, 40, 128, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        lens = torch.tensor([40, 30])
        tokens = [torch.randint(8, (2, 40, 2)), torch.randint(8, (2, 20, 2))]
        tokens[0][1, 30:] = -100
        tokens[1][1, 15:] = -100
        losses, counts = new(x, lens, tokens, [lens, lens // 2], mask=False)
        reference_losses = old(y, lens, codebook_indexes=tokens, mask=False)[:-1]
        expected = torch.stack([l.sum() for l in reference_losses])
        torch.testing.assert_close(losses, expected, rtol=0, atol=0)
        self.assertEqual(counts.tolist(), [64, 60])
        losses.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
        for n, p in new.codebook_loss_heads.named_parameters():
            if p.grad is not None:
                torch.testing.assert_close(p.grad, dict(old.codebook_loss_heads.named_parameters())[n].grad)

    def test_short_delta_is_ignored(self):
        a = args(["--num-codebooks", "2,2", "--codebook-size", "8,8", "--distillation-delta", "20,0"])
        m = MVQModel(IdentityEmbed(), IdentityEncoder(), 128, teachers(a), a).eval()
        lens = torch.tensor([10])
        loss, counts = m(torch.randn(1, 10, 128), lens,
                         [torch.zeros(1, 10, 2, dtype=torch.int64), torch.zeros(1, 5, 2, dtype=torch.int64)],
                         [lens, lens // 2], mask=False)
        self.assertEqual(loss[0].item(), 0)
        self.assertEqual(counts.tolist(), [0, 10])

    def test_masked_only_delta_uses_later_student_mask(self):
        a = args(["--num-codebooks", "2,2", "--codebook-size", "8,8", "--loss-only-mask", "1",
                  "--distillation-delta", "2,3"])
        m = MVQModel(IdentityEmbed(), IdentityEncoder(), 128, teachers(a), a).train()
        # This fake front end halves input time, as the real 50 Hz front end does.
        class Half(nn.Module):
            def forward(self, x, lens):
                return x[:, ::2], lens // 2
        m.encoder_embed = Half()
        mask = torch.zeros(1, 20)
        mask[:, 8:12] = 1
        m.apply_mask = lambda x, padding: (x, mask)
        lens = torch.tensor([20])
        _, counts = m(torch.randn(1, 20, 128), lens,
            [torch.zeros(1, 10, 2, dtype=torch.int64), torch.zeros(1, 5, 2, dtype=torch.int64)],
            [torch.tensor([10]), torch.tensor([5])])
        self.assertEqual(counts.tolist(), [2, 2])

    def test_seeded_masking_and_short_block(self):
        seed_everything(4)
        state = rng_state()
        a = compute_mask_indices((2, 100), None, 0.5, 10, require_same_masks=False)
        restore_rng(state)
        b = compute_mask_indices((2, 100), None, 0.5, 10, require_same_masks=False)
        np.testing.assert_array_equal(a, b)
        short = compute_mask_indices_block((1, 4), None, 0.5, 10, min_masks=2)
        self.assertEqual(short.shape, (1, 4))
        self.assertEqual(short.sum(), 0)

    def test_seeded_masking_reference(self):
        reference_class()
        old = importlib.import_module("model_multi_kd_w2v2_mask").compute_mask_indices
        kw = dict(shape=(2, 100), padding_mask=None, mask_prob=0.5, mask_length=10,
                  seed=3, epoch=0, indices=torch.tensor([1, 2]), require_same_masks=False)
        np.testing.assert_array_equal(compute_mask_indices(**kw), old(**kw))

    def test_mixing_offset_extension_and_uint16(self):
        a = torch.tensor([[300, 301], [302, 303], [304, 305]])
        b = torch.tensor([[700, 701], [702, 703], [704, 705]])
        self.assertEqual(mix_targets(a, b, 2, 1).tolist(), a[:2].tolist() + b.tolist())
        self.assertEqual(mix_targets(a, b, 2, 0).tolist(), a.tolist() + b[1:].tolist())

    def test_real_manifest_targets_mixing_and_cut_offsets(self):
        with tempfile.TemporaryDirectory() as d:
            manifest = create_manifest(d, count=4)
            a = args(["--train-cuts", str(manifest), "--codebook-size", "256,1024", "--mixing-prob", "1"])
            specs = teachers(a)
            cuts = CutSet.from_file(manifest)
            c = next(iter(cuts))
            validate_cut(c, specs, load_arrays=True)
            tokens = load_targets(c, specs[1])
            self.assertEqual(tokens.dtype, torch.int64)
            self.assertEqual(tokens[0, 0], 1023)
            trimmed = c.truncate(offset=0.2, duration=0.6)
            torch.testing.assert_close(load_targets(trimmed, specs[1]), tokens[5:20])
            for mode in ("max", "wavlm"):
                a.batch_mix_mode = mode
                batch = LibriSpeechDataset(a, specs, True)[cuts]
                self.assertEqual(batch["inputs"].shape[-1], 128)
                self.assertEqual(len(batch["targets"]), 2)
                self.assertEqual(batch["targets"][1].dtype, torch.int64)
            single = LibriSpeechDataset(a, specs, True)[CutSet.from_cuts([c])]
            self.assertEqual(single["cut_ids"], [c.id])

    def test_missing_required_vs_unused_teacher(self):
        with tempfile.TemporaryDirectory() as d:
            cuts = CutSet.from_file(create_manifest(d, count=1))
            c = next(iter(cuts))
            del c.dasheng_codebook_indexes
            original = parse_args(["--train-cuts", "x"])
            validate_cut(c, teachers(original), load_arrays=True)
            with self.assertRaisesRegex(ValueError, "missing required dasheng"):
                validate_cut(c, teachers(args()))
            c.wavlm_codebook_indexes.array.storage_path = str(Path(d) / "missing.h5")
            with self.assertRaisesRegex(ValueError, "wavlm.*failed loading"):
                load_targets(c, teachers(original)[0])

    def test_split_is_recording_stable(self):
        with tempfile.TemporaryDirectory() as d:
            c = next(iter(CutSet.from_file(create_manifest(d, count=1))))
            trimmed = c.truncate(offset=0.2, duration=0.6)
            self.assertEqual(validation_cut(c, 42, 0.3), validation_cut(trimmed, 42, 0.3))

    def test_validation_shards_are_disjoint_and_complete(self):
        with tempfile.TemporaryDirectory() as d:
            cuts = list(CutSet.from_file(create_manifest(d, count=6)))
            shards = [{c.id for c in cuts if validation_rank(c, r, 8)} for r in range(8)]
            self.assertEqual(set.union(*shards), {c.id for c in cuts})
            self.assertEqual(sum(map(len, shards)), len(cuts))

    def test_resume_rejects_semantic_changes(self):
        a = args()
        saved = dict(config=config_dict(a), world_size=1, manifest_sha256="x")
        a.max_iters += 10
        check_resume(saved, a, "x", 1)
        a.distillation_delta = "1,0"
        with self.assertRaises(ValueError):
            check_resume(saved, a, "x", 1)

    def test_scaling_runtime_state_roundtrip(self):
        from scaling import Balancer, Whiten
        a = small_args()
        m = build_model(a, teachers(a))
        whiten = next(x for x in m.modules() if isinstance(x, Whiten))
        balancer = next(x for x in m.modules() if isinstance(x, Balancer))
        whiten.prob = whiten.min_prob
        balancer.mem_cutoff.count = 123
        state = module_runtime_state(m)
        whiten.prob = whiten.max_prob
        balancer.mem_cutoff.count = 0
        restore_module_runtime(m, state)
        self.assertEqual(whiten.prob, whiten.min_prob)
        self.assertEqual(balancer.mem_cutoff.count, 123)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA smoke check runs on a GPU node")
    def test_backbone_encoder_parity_cuda(self):
        a = small_args()
        m = build_model(a, teachers(a)).cuda().eval()
        old = reference_class()(copy.deepcopy(m.encoder_embed), copy.deepcopy(m.encoder), encoder_dim=64,
            num_codebooks=(16, 8), teacher_frame_ratio=(1, 1), distillation_layer=(-1, -1),
            distillation_delta=(0, 0)).cuda().eval()
        x, lens = torch.randn(2, 207, 128, device="cuda"), torch.tensor([207, 157], device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out, lengths = m.forward_encoder(x, lens)
            expected, expected_lens = old.forward_encoder(x, lens)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        self.assertEqual(lengths.tolist(), [100, 75])
        torch.testing.assert_close(lengths, expected_lens)


if __name__ == "__main__":
    unittest.main()
