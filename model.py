# Copyright    2021-2023  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Wei Kang,
#                                                       Zengwei Yao)
#
# Copyright    2024 University of Cambridge      (authors: Xiaoyu Yang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Original Zipformer/MVQ architecture, simplified to offline teacher distillation."""
import torch
from torch import nn
from torch.nn import functional as F
from multi_quantization.prediction import JointCodebookLoss
from icefall.utils import make_pad_mask
from config import integers
from masking import MaskingMixin
from scaling import ScheduledFloat
from subsampling import Conv2dSubsampling
from zipformer2 import Zipformer2


def align_targets(targets, target_lens, output_lens, width, repeat):
    """Align each utterance independently; batch padding never determines frame rate."""
    if targets.dtype != torch.int64:
        raise ValueError("Targets must be signed int64 before padding/alignment")
    result = targets.new_full((len(targets), width, targets.shape[-1]), -100)
    for i in range(len(targets)):
        n, expected = int(target_lens[i]), int(output_lens[i])
        if abs(n * repeat - expected) > 5 * repeat:
            raise ValueError(f"Teacher/student length mismatch for batch item {i}: {n}*{repeat} versus {expected}")
        seq = targets[i, :n].repeat_interleave(repeat, dim=0)
        count = min(len(seq), expected)
        result[i, :count] = seq[:count]
    return result


class MVQModel(MaskingMixin, nn.Module):
    def __init__(self, encoder_embed, encoder, encoder_dim, specs, args):
        super().__init__()
        self.encoder_embed, self.encoder = encoder_embed, encoder
        self.specs = specs
        self.codebook_loss_heads = nn.ModuleList([
            JointCodebookLoss(predictor_channels=encoder_dim, num_codebooks=t.num_codebooks,
                              codebook_size=t.codebook_size, is_joint=False, reduction="none")
            for t in specs
        ])
        self.mask_emb = nn.Parameter(torch.FloatTensor(128).normal_())
        for name in ("mask_mode", "mask_prob", "mask_length", "mask_selection", "mask_other",
                     "mask_channel_prob", "mask_channel_length", "loss_only_mask"):
            setattr(self, name, getattr(args, name))
        self.min_masks = 2
        self.mask_channel_selection, self.mask_channel_other = "static", 0.0

    def forward_encoder(self, x, x_lens):
        x, x_lens = self.encoder_embed(x, x_lens)
        padding = make_pad_mask(x_lens, max_len=x.shape[1])
        out, lens = self.encoder(x.permute(1, 0, 2), x_lens, padding)
        if not torch.all(lens > 0):
            raise ValueError(f"Empty encoder output: {lens}")
        return out.permute(1, 0, 2), lens

    def forward(self, x, x_lens, codebook_indexes, target_lens, mask=True):
        mask_indices = None
        if self.training and mask:
            x, mask_indices = self.apply_mask(x.clone(), make_pad_mask(x_lens, max_len=x.shape[1]))
        out, lens = self.forward_encoder(x, x_lens)
        time_mask = None
        if self.loss_only_mask and mask_indices is not None:
            # Reference feature masking projected to the accepted 50 Hz student grid.
            time_mask = F.avg_pool1d(mask_indices.float().unsqueeze(1), 2).squeeze(1) >= 0.5
            time_mask = time_mask[:, :out.shape[1]]
        losses, counts = [], []
        for t, head, tokens, lengths in zip(self.specs, self.codebook_loss_heads, codebook_indexes, target_lens):
            if tokens is None or not t.active:
                losses.append(out.sum() * 0)
                counts.append(out.new_zeros(()))
                continue
            aligned = align_targets(tokens, lengths, lens, out.shape[1], t.repeat)
            d = t.delta
            if d >= out.shape[1]:
                losses.append(out.sum() * 0)
                counts.append(out.new_zeros(()))
                continue
            prediction = out[:, d:]
            aligned = aligned[:, :out.shape[1] - d]
            valid = ~make_pad_mask((lens - d).clamp(min=0), max_len=prediction.shape[1])
            if time_mask is not None:
                valid &= time_mask[:, d:]
            aligned = aligned.masked_fill(~valid.unsqueeze(-1), -100)
            ce = head(prediction.float(), aligned).reshape(*aligned.shape)
            # Original normalization: average codebooks, sum frames and utterances.
            losses.append((ce.sum(dim=2) / t.num_codebooks).sum(dim=1).sum())
            counts.append((aligned >= 0).all(dim=-1).sum().to(out.dtype))
        return torch.stack(losses), torch.stack(counts)


def build_model(args, specs):
    embed = Conv2dSubsampling(in_channels=128, out_channels=integers(args.encoder_dim)[0],
                             dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)))
    encoder = Zipformer2(
        output_downsampling_factor=1, downsampling_factor=integers(args.downsampling_factor),
        num_encoder_layers=integers(args.num_encoder_layers), encoder_dim=integers(args.encoder_dim),
        encoder_unmasked_dim=integers(args.encoder_unmasked_dim), query_head_dim=integers(args.query_head_dim),
        pos_head_dim=integers(args.pos_head_dim), value_head_dim=integers(args.value_head_dim),
        pos_dim=args.pos_dim, num_heads=integers(args.num_heads), feedforward_dim=integers(args.feedforward_dim),
        cnn_module_kernel=integers(args.cnn_module_kernel),
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)), warmup_batches=4000.0,
        causal=False, chunk_size=(-1,), left_context_frames=(-1,))
    return MVQModel(embed, encoder, max(integers(args.encoder_dim)), specs, args)


def set_batch_count(model, batch_count):
    for name, module in model.named_modules():
        if hasattr(module, "batch_count"):
            module.batch_count = batch_count
        if hasattr(module, "name"):
            module.name = name
