# Copyright 2024 University of Cambridge (authors: Xiaoyu Yang)
# Licensed under the Apache License, Version 2.0; see LICENSE.
"""SPEAR-style MVQ pre-training on one prepared LibriSpeech manifest."""
import copy
import json
import logging
import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from icefall.checkpoint import update_averaged_model
from icefall.utils import AttributeDict, get_parameter_groups_with_lrs
from config import parse_args, teachers
from data import load_splits, make_loader, make_validation_loader
from model import build_model, set_batch_count
from optim import ScaledAdam, Eden
from utils import (seed_everything, rng_state, restore_rng, config_dict, manifest_hash,
                   check_resume, save_training, restore_module_runtime, configure_determinism)


def reduce_sum(value, world_size):
    if world_size > 1:
        dist.all_reduce(value)
    return value


def move_batch(batch, device):
    return (batch["inputs"].to(device), batch["input_lens"].to(device),
            [x.to(device) if x is not None else None for x in batch["targets"]],
            [x.to(device) if x is not None else None for x in batch["target_lens"]])


def report(prefix, totals, specs, step, writer):
    values = {}
    for i, t in enumerate(specs):
        loss, frames = totals[i], totals[i + len(specs)]
        if frames > 0:
            values[t.name + "_loss_per_frame"] = float(loss / frames)
            values[t.name + "_frames"] = int(frames)
    logging.info("%s step=%d %s", prefix, step, values)
    if writer:
        for key, value in values.items():
            writer.add_scalar(prefix + "/" + key, value, step)


@torch.no_grad()
def validate(model, loader, specs, dtype, device, world_size):
    # Preserve training RNG so validation cadence does not change augmentation.
    state = rng_state()
    model.eval()
    totals = torch.zeros(2 * len(specs), device=device, dtype=torch.float64)
    try:
        for batch in (() if loader is None else loader):
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                losses, counts = model(*move_batch(batch, device), mask=False)
            totals += torch.cat((losses.double(), counts.double()))
        reduce_sum(totals, world_size)
        if not torch.isfinite(totals).all() or totals[len(specs):].sum() == 0:
            raise ValueError("Validation has no usable supervision or nonfinite loss")
        return totals
    finally:
        restore_rng(state)
        model.train()


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA; use the CPU contract checks on login nodes")
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    rank, local_rank, world_size = (int(os.environ.get(k, default)) for k, default in
                                   (("RANK", 0), ("LOCAL_RANK", 0), ("WORLD_SIZE", 1)))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group("nccl")
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s rank={rank} %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.exp_dir / f"train-rank{rank}.log")])
    writer = SummaryWriter(str(args.exp_dir / "tensorboard")) if rank == 0 and args.tensorboard else None
    try:
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("GPU does not support bf16; select --precision fp16 or fp32")
        specs = teachers(args)
        seed_everything(args.seed)
        model = build_model(args, specs).to(device)
        configure_determinism(model, args.deterministic)
        model_avg = copy.deepcopy(model).to(device="cpu", dtype=torch.float64) if rank == 0 else None
        if world_size > 1:
            model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        raw_model = model.module if world_size > 1 else model
        optimizer = ScaledAdam(get_parameter_groups_with_lrs(raw_model, lr=args.base_lr, include_names=True),
                               lr=args.base_lr, clipping_scale=2.0)
        scheduler = Eden(optimizer, args.lr_batches, args.lr_hours, warmup_batches=args.warmup_batches)
        scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16, init_scale=1.0)
        train_cuts, valid_cuts = load_splits(args)
        train_loader, train_generator = make_loader(train_cuts, args, specs, True, rank, world_size)
        valid_loader, valid_generator = (make_validation_loader(valid_cuts, args, specs, rank, world_size)
                                         if valid_cuts is not None else (None, None))
        fingerprint = [manifest_hash(args.train_cuts) if rank == 0 else None]
        if world_size > 1:
            dist.broadcast_object_list(fingerprint)
        fingerprint = fingerprint[0]
        step, start_epoch, audio_hours, restored = 0, 0, 0.0, False
        if args.resume:
            # Trusted local training checkpoints contain Python/NumPy RNG and optimizer objects.
            saved = torch.load(args.resume, map_location="cpu", weights_only=False)
            check_resume(saved, args, fingerprint, world_size)
            raw_model.load_state_dict(saved["model"], strict=True)
            if rank == 0:
                model_avg.load_state_dict(saved["model_avg"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            scaler.load_state_dict(saved["grad_scaler"])
            step, start_epoch, audio_hours = saved["global_step"], saved["epoch"], saved["audio_hours"]
            state = saved["rank_states"][rank]
            restore_module_runtime(raw_model, state["module_runtime"])
            train_loader.sampler.load_state_dict(state["sampler"])
            train_generator.set_state(state["loader_generator"])
            if valid_generator is not None:
                valid_generator.set_state(state["valid_generator"])
            restore_rng(state["rng"])
            restored = True
            # Set names for diagnostic logs; runtime attributes were restored separately.
            previous = max(0, ((step - 1) // 10) * 10)
            set_batch_count(raw_model, previous * args.max_duration * world_size / args.ref_duration)
            logging.info("Resumed step=%d epoch=%d", step, start_epoch)
        if rank == 0:
            (args.exp_dir / "config.json").write_text(json.dumps(config_dict(args), indent=2) + "\n")
        logging.info("Parameters=%d; student=50 Hz; teachers=%s", sum(p.numel() for p in raw_model.parameters()), specs)
        weights = torch.tensor([t.weight if t.active else 0 for t in specs], device=device)
        total = torch.zeros(2 * len(specs), device=device, dtype=torch.float64)
        last_saved = None
        for epoch in range(start_epoch, args.num_epochs):
            if step >= args.max_iters:
                break
            if not restored:
                seed_everything(args.seed + epoch)
                train_loader.sampler.set_epoch(epoch)
            restored = False
            batches = 0
            for batch in train_loader:
                if step >= args.max_iters:
                    break
                if step % 10 == 0:
                    set_batch_count(raw_model, step * args.max_duration * world_size / args.ref_duration)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                    losses, counts = model(*move_batch(batch, device))
                    loss = (weights * losses).sum()
                usable = reduce_sum(counts.detach().double().sum().reshape(1), world_size)
                finite_loss = torch.isfinite(loss).int()
                if world_size > 1:
                    dist.all_reduce(finite_loss, op=dist.ReduceOp.MIN)
                if usable.item() == 0 or not finite_loss.item():
                    raise ValueError(f"Nonfinite loss or no supervision at step {step + 1}; cuts={batch['cut_ids']}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grads = [p.grad for p in raw_model.parameters() if p.grad is not None]
                finite = torch.stack([torch.isfinite(g).all() for g in grads]).all().int()
                if world_size > 1:
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item() and dtype != torch.float16:
                    raise FloatingPointError(f"Nonfinite gradient at step {step + 1}")
                step += 1
                batches += 1
                scheduler.step_batch(step)
                scheduler.step_epoch(step * args.max_duration * world_size / 3600)
                scaler.step(optimizer)
                scaler.update()
                duration = reduce_sum(torch.tensor(batch["duration"], device=device, dtype=torch.float64), world_size)
                audio_hours += duration.item() / 3600
                total += torch.cat((losses.detach().double(), counts.detach().double()))
                # Do not retain old autograd graphs/gradient lists into the next forward.
                # The source Balancer's memory-dependent cutoff otherwise sees different
                # allocations in uninterrupted and freshly resumed processes.
                del losses, counts, loss, grads
                if rank == 0 and step % args.average_period == 0:
                    update_averaged_model(AttributeDict(average_period=args.average_period, batch_idx_train=step),
                                          raw_model, model_avg)
                if step % args.log_interval == 0 or step == 1:
                    logged = reduce_sum(total.clone(), world_size)
                    report("train", logged, specs, step, writer)
                    logging.info("lr=%.6g audio_hours=%.4f", max(scheduler.get_last_lr()), audio_hours)
                    total.zero_()
                if valid_cuts is not None and (step % args.valid_interval == 0 or step == 1):
                    val = validate(raw_model, valid_loader, specs, dtype, device, world_size)
                    report("valid", val, specs, step, writer)
                if step % args.save_every_n == 0 or step >= args.max_iters:
                    path = args.exp_dir / f"checkpoint-{step}.pt"
                    save_training(path, raw_model, model_avg, optimizer, scheduler, scaler, train_loader,
                                  train_generator, valid_generator, args, step, epoch, fingerprint,
                                  audio_hours, rank, world_size)
                    last_saved = step
                if step >= args.max_iters:
                    break
            if batches == 0 and epoch == start_epoch and not args.resume:
                raise ValueError("No training batches")
            if step > 0 and last_saved != step:
                save_training(args.exp_dir / f"checkpoint-{step}.pt", raw_model, model_avg, optimizer,
                              scheduler, scaler, train_loader, train_generator, valid_generator, args,
                              step, epoch, fingerprint, audio_hours, rank, world_size)
                last_saved = step
            if step >= args.max_iters:
                break
        logging.info("Finished at step=%d", step)
    finally:
        if writer:
            writer.close()
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    run(parse_args())
