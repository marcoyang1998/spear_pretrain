"""Configuration for one LibriSpeech manifest and two offline MVQ teachers."""
import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path


def integers(value):
    return tuple(int(x) for x in value.split(",")) if isinstance(value, str) else tuple(value)


def boolean(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() in ("1", "true", "yes"):
        return True
    if str(value).lower() in ("0", "false", "no"):
        return False
    raise argparse.ArgumentTypeError("Expected true/false or 1/0")


@dataclass(frozen=True)
class Teacher:
    name: str
    field: str
    num_codebooks: int
    codebook_size: int
    frame_rate: int
    delta: int
    weight: float
    active: bool

    @property
    def repeat(self):
        return 50 // self.frame_rate


def teachers(args):
    pairs = [args.teacher_fields.split(","), integers(args.num_codebooks),
             integers(args.codebook_size), integers(args.teacher_frame_rate),
             integers(args.distillation_delta),
             tuple(float(x) for x in args.teacher_loss_weights.split(","))]
    if any(len(x) != 2 for x in pairs):
        raise ValueError("Each teacher option must contain exactly two comma-separated values")
    active = (True, args.teacher_routing == "both")
    if args.teacher_routing == "teacher2":
        active = (False, True)
    result = []
    for i, name in enumerate(("wavlm", "dasheng")):
        field, cb, size, rate, delta, weight = (p[i] for p in pairs)
        if not field or cb < 2 or size < 2 or rate not in (25, 50) or delta < 0:
            raise ValueError(f"Invalid metadata for {name}: require >=2 codebooks/entries, 25/50 Hz, delta>=0")
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Teacher weights must be finite and nonnegative")
        result.append(Teacher(name, field, cb, size, rate, delta, weight, active[i] and weight > 0))
    if not any(t.active for t in result):
        raise ValueError("At least one teacher must have positive weight and be enabled")
    if pairs[0][0] == pairs[0][1]:
        raise ValueError("The two teacher fields must have distinct names")
    return tuple(result)


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", type=Path, help="JSON defaults; command-line options override them")
    p.add_argument("--train-cuts", type=Path, help="One prepared LibriSpeech CutSet jsonl.gz")
    p.add_argument("--exp-dir", type=Path, default=Path("exp"))
    p.add_argument("--resume", type=Path)
    p.add_argument("--teacher-routing", choices=("original", "both", "teacher1", "teacher2"), default="original")
    p.add_argument("--teacher-fields", default="wavlm_codebook_indexes,dasheng_codebook_indexes")
    p.add_argument("--num-codebooks", default="16,8")
    p.add_argument("--codebook-size", default="256,256")
    p.add_argument("--teacher-frame-rate", default="50,25")
    p.add_argument("--distillation-delta", default="0,0")
    p.add_argument("--teacher-loss-weights", default="1.0,0.1")
    p.add_argument("--validation-fraction", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-duration", type=float, default=200)
    p.add_argument("--max-cuts", type=int, default=200)
    p.add_argument("--num-buckets", type=int, default=20)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--min-cut-duration", type=float, default=0.9)
    p.add_argument("--max-cut-duration", type=float, default=32)
    p.add_argument("--batch-duration-factor", type=float, default=5)
    p.add_argument("--batch-mixing", type=boolean, default=True)
    p.add_argument("--token-mixing", type=boolean, default=True)
    p.add_argument("--mixing-prob", type=float, default=0.2)
    p.add_argument("--batch-mix-mode", choices=("wavlm", "max"), default="max")
    p.add_argument("--mix-delay-max", type=float, default=2)
    p.add_argument("--min-snr", type=float, default=-5)
    p.add_argument("--max-snr", type=float, default=5)
    p.add_argument("--mask-mode", choices=("w2v2", "block"), default="w2v2")
    p.add_argument("--mask-prob", type=float, default=0.65)
    p.add_argument("--mask-length", type=int, default=10)
    p.add_argument("--mask-selection", choices=("static", "uniform", "normal", "poisson"), default="static")
    p.add_argument("--mask-other", type=float, default=0)
    p.add_argument("--mask-channel-prob", type=float, default=0.25)
    p.add_argument("--mask-channel-length", type=int, default=15)
    p.add_argument("--loss-only-mask", type=boolean, default=False)
    defaults = dict(num_encoder_layers="2,2,3,4,3,2", downsampling_factor="1,2,4,8,4,2",
                    feedforward_dim="512,768,1024,1536,1024,768", num_heads="4,4,4,8,4,4",
                    encoder_dim="192,256,384,512,384,256", query_head_dim="32",
                    value_head_dim="12", pos_head_dim="4", encoder_unmasked_dim="192,192,256,256,256,192",
                    cnn_module_kernel="31,31,15,15,15,31")
    for name, value in defaults.items():
        p.add_argument("--" + name.replace("_", "-"), default=value)
    p.add_argument("--pos-dim", type=int, default=48)
    p.add_argument("--output-downsampling-factor", type=int, choices=(1,), default=1)
    p.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--deterministic", type=boolean, default=False,
                   help="Deterministic CUDA kernels and no allocation-dependent Balancer skipping")
    p.add_argument("--base-lr", type=float, default=0.045)
    p.add_argument("--warmup-batches", type=float, default=500)
    p.add_argument("--lr-batches", type=float, default=7500)
    p.add_argument("--lr-hours", type=float, default=30000)
    p.add_argument("--ref-duration", type=float, default=600)
    p.add_argument("--num-epochs", type=int, default=30)
    p.add_argument("--max-iters", type=int, default=200000)
    p.add_argument("--save-every-n", type=int, default=4000)
    p.add_argument("--keep-last-k", type=int, default=30)
    p.add_argument("--average-period", type=int, default=200)
    p.add_argument("--valid-interval", type=int, default=3000)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--tensorboard", type=boolean, default=True)
    return p


def parse_args(argv=None):
    p = parser()
    pre, _ = p.parse_known_args(argv)
    if pre.config:
        defaults = json.loads(pre.config.read_text())
        actions = {a.dest: a for a in p._actions}
        unknown = set(defaults) - set(actions)
        if unknown:
            p.error(f"Unknown config keys: {sorted(unknown)} (use underscore names)")
        converted = {}
        for k, v in defaults.items():
            a = actions[k]
            converted[k] = a.type(v) if a.type else v
            if a.choices and converted[k] not in a.choices:
                p.error(f"Invalid config value for {k}: {v}")
        p.set_defaults(**converted)
    args = p.parse_args(argv)
    if args.train_cuts is None:
        p.error("--train-cuts is required")
    if not 0 <= args.validation_fraction < 1:
        p.error("validation-fraction must be in [0,1)")
    for name in ("max_duration", "max_cuts", "num_buckets", "batch_duration_factor", "mask_length",
                 "mask_channel_length", "base_lr", "lr_batches", "lr_hours", "ref_duration", "num_epochs",
                 "max_iters", "save_every_n", "keep_last_k", "average_period", "valid_interval", "log_interval"):
        v = getattr(args, name)
        if not math.isfinite(v) or v <= 0:
            p.error(f"{name} must be positive and finite")
    if args.num_workers < 0 or args.min_cut_duration <= 0 or args.max_cut_duration < args.min_cut_duration:
        p.error("Invalid worker count or cut-duration range")
    if args.min_snr > args.max_snr or args.mix_delay_max < 0:
        p.error("Invalid mixing SNR/delay range")
    for name in ("mask_prob", "mask_channel_prob", "mixing_prob"):
        if not 0 <= getattr(args, name) <= 1:
            p.error(f"{name} must be in [0,1]")
    if args.save_every_n < args.average_period:
        p.error("save-every-n must be >= average-period")
    if args.loss_only_mask and args.mask_prob == 0:
        p.error("masked-only loss requires nonzero time masking")
    teachers(args)
    return args
