# Copyright 2026 University of Cambridge (author: Xiaoyu Yang)
# Licensed under the Apache License, Version 2.0; see ../LICENSE.
"""Local-only SPEAR XLarge v2 model loading."""
import torch
from .spear_hf.modeling_spear import SpearModel


class SpearTeacher(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model, info = SpearModel.from_pretrained(
            model_path, local_files_only=True, output_loading_info=True)
        if any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
            raise ValueError(f'SPEAR checkpoint did not load exactly: {info}')
        self.eval()

    @torch.no_grad()
    def forward(self, audio, audio_lens, layer=-1):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            outputs = self.model(audio, audio_lens)
        embeddings = outputs['encoder_out'] if layer == -1 else outputs['hidden_states'][layer]
        return embeddings.float(), outputs['encoder_out_lens']
