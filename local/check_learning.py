"""CUDA fixed-batch learning check with both teacher heads and original ScaledAdam."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from config import parse_args, teachers
from model import build_model
from optim import ScaledAdam
from icefall.utils import get_parameter_groups_with_lrs
from utils import seed_everything


if __name__ == "__main__":
    seed_everything(42)
    a = parse_args(["--config", "configs/smoke.json", "--train-cuts", "unused"])
    specs = teachers(a)
    m = build_model(a, specs).cuda().eval()
    optimizer = ScaledAdam(get_parameter_groups_with_lrs(m, lr=0.01, include_names=True), lr=0.01, clipping_scale=2)
    x = torch.randn(2, 207, 128, device="cuda")
    lens = torch.tensor([207, 207], device="cuda")
    targets = [torch.randint(t.codebook_size, (2, 100 // t.repeat, t.num_codebooks), device="cuda") for t in specs]
    lengths = [torch.tensor([len(y[0])] * 2, device="cuda") for y in targets]
    weights = torch.tensor([t.weight for t in specs], device="cuda")
    values = []
    for step in range(9):
        optimizer.zero_grad()
        losses, counts = m(x, lens, targets, lengths, mask=False)
        loss = (losses * weights).sum()
        values.append(loss.item())
        assert torch.isfinite(loss)
        if step < 8:
            loss.backward()
            for head in m.codebook_loss_heads:
                assert head.linear2b_weight.grad is not None
                assert torch.isfinite(head.linear2b_weight.grad).all()
            optimizer.step()
    assert values[-1] < values[0], values
    print(f"Both teacher heads learn: fixed-batch loss {values[0]:.4f} -> {values[-1]:.4f}")
