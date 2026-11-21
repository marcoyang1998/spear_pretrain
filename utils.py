"""Reproducible state and atomic checkpoints for the extracted trainer."""
import hashlib
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from scaling import Balancer, Whiten


class NoMemorySkip:
    def __call__(self, allocated_bytes):
        return False


def configure_determinism(model, enabled):
    if enabled:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        for module in model.modules():
            if isinstance(module, Balancer):
                # Keep Balancer's scheduled probability/gradient transformation;
                # disable only its memory-pressure skip heuristic for reproducibility.
                module.mem_cutoff = NoMemorySkip()


def module_runtime_state(model):
    """Original scaling layers have training state outside their state_dict."""
    state = {}
    for name, module in model.named_modules():
        values = {}
        if isinstance(module, Whiten):
            values["prob"] = module.prob
        if isinstance(module, Balancer):
            values["mem_cutoff"] = vars(module.mem_cutoff).copy()
        if hasattr(module, "pe") and isinstance(module.pe, torch.Tensor):
            values["pe"] = module.pe.detach().cpu().clone()
        if hasattr(module, "batch_count"):
            values["batch_count"] = module.batch_count
        if values:
            state[name] = values
    return state


def restore_module_runtime(model, state):
    modules = dict(model.named_modules())
    device = next(model.parameters()).device
    for name, values in state.items():
        module = modules[name]
        for key, value in values.items():
            if key == "mem_cutoff":
                vars(module.mem_cutoff).update(value)
            else:
                setattr(module, key, value.to(device) if isinstance(value, torch.Tensor) else value)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def config_dict(args):
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}


def manifest_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_resume(saved, args, fingerprint, world_size):
    if saved["world_size"] != world_size:
        raise ValueError("Resume requires the original world size (sampler/rank RNG state)")
    if saved["manifest_sha256"] != fingerprint:
        raise ValueError("Input manifest changed since the checkpoint")
    runtime = {"config", "train_cuts", "exp_dir", "resume", "num_epochs", "max_iters",
               "save_every_n", "keep_last_k", "log_interval", "tensorboard"}
    current = config_dict(args)
    changed = [k for k in saved["config"] if k not in runtime and saved["config"][k] != current.get(k)]
    if changed:
        raise ValueError(f"Resume would change training semantics: {changed}")


def save_training(path, model, model_avg, optimizer, scheduler, scaler, loader, loader_generator,
                  valid_generator, args, step, epoch, fingerprint, audio_hours, rank, world_size):
    local = dict(rng=rng_state(), module_runtime=module_runtime_state(model), sampler=loader.sampler.state_dict(),
                 loader_generator=loader_generator.get_state(),
                 valid_generator=valid_generator.get_state() if valid_generator else None)
    states = [None] * world_size
    if world_size > 1:
        dist.all_gather_object(states, local)
    else:
        states[0] = local
    if rank == 0:
        checkpoint = dict(format_version=1, model=model.state_dict(),
            model_avg=model_avg.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), grad_scaler=scaler.state_dict(),
            rank_states=states, config=config_dict(args), global_step=step, epoch=epoch,
            manifest_sha256=fingerprint, world_size=world_size, audio_hours=audio_hours)
        temporary = path.with_suffix(".tmp")
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
        # Only prune this project's numbered checkpoints; leave unrelated files untouched.
        checkpoints = sorted(path.parent.glob("checkpoint-*.pt"), key=lambda p: int(p.stem.split("-")[-1]))
        for old in checkpoints[:-args.keep_last_k]:
            old.unlink()
    if world_size > 1:
        dist.barrier()
