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


"""Original masking helpers; seeded RNG and short-block fixes are documented in PROVENANCE.md."""
import logging
import random
from typing import Optional, Tuple
import numpy as np
import torch

def index_put(tensor, indices, value):
    tensor[indices] = value
    return tensor    

def compute_mask_indices_block(
    shape,
    padding_mask,
    mask_prob: float = 0.5,
    mask_length: int = 10,
    min_masks: int = 2,
):
    # self-implemented mask, no overlap
    B,T = shape
    mask_indices = []
    for i in range(B):
        if padding_mask is not None:
            num_segments = (T - padding_mask[i].sum()) // mask_length # discard the last few frames
        else:
            num_segments = T // mask_length 
        num_segments = int(num_segments)
        required = min(min_masks, num_segments) if mask_prob > 0 else 0
        segment_mask = torch.rand(num_segments) < mask_prob 
        while sum(segment_mask) < required:
            segment_mask = torch.rand(num_segments) < mask_prob
        segment_mask_expanded = segment_mask.unsqueeze(-1).expand(num_segments, mask_length)
        segment_mask_expanded = segment_mask_expanded.reshape(-1).float()
        if segment_mask_expanded.size(0) < T:
            pad = T - segment_mask_expanded.size(0)
            segment_mask_expanded = torch.cat([segment_mask_expanded, torch.zeros(pad)])
        mask_indices.append(segment_mask_expanded)

    mask_indices = torch.stack(mask_indices)
    return mask_indices

def compute_mask_indices(
    shape: Tuple[int, int],
    padding_mask: Optional[torch.Tensor],
    mask_prob: float,
    mask_length: int,
    mask_type: str = "static",
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 0,
    require_same_masks: bool = True,
    mask_dropout: float = 0.0,
    add_masks: bool = False,
    seed: Optional[int] = None,
    epoch: Optional[int] = None,
    indices: Optional[torch.Tensor] = None,
    idc_select_ver: int = 1,  # 2 to reproduce mask_tokens_dataset
    num_mask_ver: int = 2,  # 2 to reproduce mask_tokens_dataset
) -> np.ndarray:
    """
    Computes random mask spans for a given shape

    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
            poisson = sample from possion distribution with lambda = mask length
        min_masks: minimum number of masked spans
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
        require_same_masks: if true, will randomly drop out masks until same amount of masks remains in each sample
        mask_dropout: randomly dropout this percentage of masks in each example
    """

    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    if num_mask_ver == 1:
        all_num_mask = int(
            # add a random number for probabilistic rounding
            mask_prob * all_sz / float(mask_length)
            + np.random.rand()
        )
        all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(bsz):
        if seed is not None and epoch is not None and indices is not None:
            seed_i = int(hash((seed, epoch, indices[i].item())) % 1e6)
        else:
            seed_i = int(np.random.randint(0, 2**32 - 1))

        rng = np.random.default_rng(seed_i)

        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            assert sz >= 0, sz
        else:
            sz = all_sz

        if num_mask_ver == 1:
            if padding_mask is not None:
                num_mask = int(
                    # add a random number for probabilistic rounding
                    mask_prob * sz / float(mask_length)
                    + np.random.rand()
                )
                num_mask = max(min_masks, num_mask)
            else:
                num_mask = all_num_mask
        elif num_mask_ver == 2:
            num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * sz / float(mask_length)
                + rng.random()
            )
            num_mask = max(min_masks, num_mask)
            hard_max = sz // mask_length
            num_mask = min(hard_max, num_mask) # prevent whole sequence being masked
        else:
            raise ValueError()

        if num_mask == 0:
            mask_idcs.append(np.array([], dtype=int))
            continue

        if mask_type == "static":
            lengths = np.full(num_mask, mask_length)
        elif mask_type == "uniform":
            lengths = rng.integers(mask_other, mask_length * 2 + 1, size=num_mask)
        elif mask_type == "normal":
            lengths = rng.normal(mask_length, mask_other, size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == "poisson":
            lengths = rng.poisson(mask_length, size=num_mask)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise Exception("unknown mask selection " + mask_type)

        if sum(lengths) == 0:
            if mask_type == "static":
                raise ValueError("this should never happens")
            else:
                lengths = [min(mask_length, sz - 1)]

        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = rng.integers(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.fromiter(
                    (e - s if e - s >= length + min_space else 0 for s, e in parts),
                    int,
                )
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = rng.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            if idc_select_ver == 1:
                min_len = min(lengths)
                if sz - min_len <= num_mask:
                    min_len = sz - num_mask - 1
                mask_idc = rng.choice(sz - min_len, num_mask, replace=False)
            elif idc_select_ver == 2:
                mask_idc = rng.choice(sz, num_mask, replace=False)
            else:
                raise ValueError()

            mask_idc = np.asarray(
                [
                    mask_idc[j] + offset
                    for j in range(len(mask_idc))
                    for offset in range(lengths[j])
                ]
            )

        mask_idc = np.unique(mask_idc[mask_idc < sz])
        if len(mask_idc) >= sz:
            
            raise ValueError(
                (
                    f"the entire sequence is masked. "
                    f"sz={sz}; mask_idc[mask_idc]; "
                    f"index={indices[i] if indices is not None else None}"
                )
            )
        mask_idcs.append(mask_idc)

    target_len = None
    if require_same_masks:
        if add_masks:
            target_len = max([len(m) for m in mask_idcs])
        else:
            target_len = min([len(m) for m in mask_idcs])

    for i, mask_idc in enumerate(mask_idcs):
        if target_len is not None and len(mask_idc) > target_len:
            mask_idc = rng.choice(mask_idc, target_len, replace=False)

        mask[i, mask_idc] = True

        if target_len is not None and len(mask_idc) < target_len:
            unmasked = np.flatnonzero(~mask[i])
            to_mask = rng.choice(unmasked, target_len - len(mask_idc), replace=False)
            mask[i, to_mask] = True

        if mask_dropout > 0:
            masked = np.flatnonzero(mask[i])
            num_holes = np.rint(len(masked) * mask_dropout).astype(int)
            to_drop = rng.choice(masked, num_holes, replace=False)
            mask[i, to_drop] = False

    return mask

class MaskingMixin:
    def apply_mask(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply mask according to the mask_mode, return the masked features and the masked positions

        Args:
            x (torch.Tensor): The input fbank features
            padding_mask (torch.Tensor, optional): The padding mask

        Returns:
            The masked fbank feature and the masked_indices, with masked positions as 1
        """
        # apply mask to the fbank features, two modes applicable
        if self.mask_mode == "w2v2":
            x, masked_indices = self.apply_mask_w2v2(x, padding_mask)
        elif self.mask_mode == "block":
            x, masked_indices = self.apply_mask_block(x, padding_mask)
        else:
            raise NotImplementedError()
        
        if random.random() > 0.97:
            logging.info(f"Apply {self.mask_mode} masking. A proportion of {masked_indices.sum()/masked_indices.numel():.2f} frames are masked")
        return x, masked_indices

    def apply_mask_block(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor = None
    ):
        B,T,C = x.shape
        if self.mask_prob == 0.0:
            return x, torch.zeros((B, T), device=x.device)

        mask_indices = compute_mask_indices_block(
            shape=(B,T),
            padding_mask=padding_mask,
            mask_prob=self.mask_prob,
            mask_length=self.mask_length,
            min_masks=self.min_masks,
        ).to(x.device)
        
        x = index_put(x, mask_indices.bool(), self.mask_emb)

        return x, mask_indices

    def apply_mask_w2v2(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor = None
    ):
        # this function is modified from fairseq: https://github.com/facebookresearch/fairseq/blob/bedb259bf34a9fc22073c13a1cee23192fa70ef3/fairseq/models/wav2vec/wav2vec2.py#L429
        # The masked indices have value 1
        B, T, C = x.shape
        
        # we mask channel first, then mask timestamps
        if self.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=False,
                min_space=1,
                require_same_masks=False,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            if random.random() > 0.98:
                logging.info(f"A proportion of {mask_channel_indices.sum()/mask_channel_indices.numel():.2f} feature dims are masked")
            x[mask_channel_indices] = 0

        if self.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob,
                self.mask_length,
                mask_type=self.mask_selection,
                mask_other=self.mask_other,
                min_masks=2, # fixed
                no_overlap=False,  # False
                min_space=1,  # 1
                require_same_masks=False,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x = index_put(x, mask_indices, self.mask_emb)
            mask_indices = mask_indices.float()
        else:
            mask_indices = None

        return x, mask_indices
