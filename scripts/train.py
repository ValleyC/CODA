"""
Training script for Selection-Based Hierarchical Score Following.

Two-phase curriculum:
  Phase 1 (GT routing): Pure teacher forcing. System logits scored over ALL
    systems; bar/note stages use GT system/bar routing.
  Phase 2 (Scheduled sampling): Gradually replace GT system routing with
    predicted routing (--scheduled_sampling). Bar targets are remapped to
    the predicted system's candidate set. If the GT bar is not in the
    predicted system, bar/note loss is masked (system loss always trained).

Usage:
    # Phase 1
    python scripts/train.py \
        --config configs/coda.yaml \
        --train_sets path/to/train_data --val_sets path/to/val_data \
        --tag selection_v2_phase1 --temporal_priors --augment

    # Phase 2 (fine-tune from Phase 1 best checkpoint)
    python scripts/train.py \
        --config configs/coda.yaml \
        --train_sets path/to/train_data --val_sets path/to/val_data \
        --param_path params/PHASE1_RUN/best_checkpoint.pt \
        --tag selection_v2_phase2 --temporal_priors --augment \
        --scheduled_sampling --ss_max_p 0.7 --ss_ramp_epochs 5
"""

import argparse
from contextlib import nullcontext
import json
import math
import multiprocessing
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from torch.utils.data import Dataset, DataLoader, SequentialSampler, BatchSampler
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from time import gmtime, strftime

from coda.utils.dist_utils import (
    init_distributed_mode, is_main_process, get_rank, reduce_dict
)
from coda.utils.general import (
    AverageMeter,
    isolated_python_numpy_torch_seed,
    load_yaml,
    stable_named_seed,
)
from coda.dataset import (
    load_dataset, selection_collate_wrapper, selection_getitem
)
from coda.utils.checkpoint import (
    atomic_torch_save,
    extract_model_state,
    make_training_checkpoint,
    restore_rng_state,
    validate_resume_checkpoint,
)
from coda.utils.loss import selection_loss
from coda.utils.optim import (
    build_adamw,
    guarded_optimizer_step,
)
from coda.utils.sampling import (
    EpochSeededRandomSampler,
    epoch_sample_seed,
    training_phase_seed,
)
from coda.utils.streaming_eval import streaming_eval


class SelectionDatasetWrapper(Dataset):
    """Wraps SequenceDataset to add page layout metadata via selection_getitem."""

    def __init__(self, dataset, augmentation_seed=None):
        self.dataset = dataset
        self.augmentation_seed = augmentation_seed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        if isinstance(item, tuple):
            if len(item) != 2 or self.augmentation_seed is None:
                raise ValueError("Epoch-tagged indices require an augmentation seed")
            epoch, item = map(int, item)
            sample_seed = epoch_sample_seed(self.augmentation_seed, epoch, item)
            # Persistent worker RNG streams are not part of a model
            # checkpoint. Derive every augmentation locally so restarting at
            # an epoch boundary reproduces the same stochastic sample without
            # perturbing the worker's caller state.
            with isolated_python_numpy_torch_seed(sample_seed):
                return selection_getitem(self.dataset, item)
        return selection_getitem(self.dataset, item)


def iterate_selection(network, dataloader, criterion, optimizer=None,
                      clip_grads=None, device=torch.device('cuda'),
                      tempo_aug=False, note_weight=1.0, label_smoothing=0.0,
                      epoch=0, temporal_priors=False, p_pred=0.0,
                      amp=False, amp_dtype=torch.float16, scaler=None,
                      max_batches=None, max_nonfinite_grad_skips=8):
    """
    Training/validation loop for SelectionCascadeModel.

    With p_pred=0: pure GT routing (Phase 1).
    With p_pred>0: scheduled sampling — some samples route through predicted
    system. Bar targets are remapped; if GT bar not in predicted system,
    bar/note loss is masked (system loss always computed).
    """
    train = optimizer is not None
    losses = {}
    recent_losses = {}
    skipped_nonfinite_grad_batches = 0
    recovered_grad_norm_overflows = 0

    mode_str = "Train" if train else "Val"
    if is_main_process():
        progress_total = len(dataloader)
        if max_batches is not None:
            progress_total = min(progress_total, max_batches)
        # ``set_postfix`` refreshes immediately by default, which previously
        # forced one terminal (and redirected-log) write per batch.  Full-scale
        # epochs contain more than 50k batches, so that produced very large
        # logs and measurable Python/I/O overhead.  Keep the counters current
        # on every step but render at a human-scale cadence.
        progress_bar = tqdm(
            total=progress_total,
            ncols=140,
            desc=f"{mode_str} E{epoch}",
            mininterval=5.0,
        )

    for batch_idx, data in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        if train:
            optimizer.zero_grad(set_to_none=True)

        scores = data.scores.to(device, non_blocking=True)
        perf = [p.to(device, non_blocking=True) for p in data.perf]
        gt_system_idx = data.gt_system_idx.to(device, non_blocking=True)
        gt_bar_in_sys = data.gt_bar_in_sys.to(device, non_blocking=True)
        gt_note_position = data.gt_note_position.to(device, non_blocking=True)
        gt_valid = data.gt_valid.to(device, non_blocking=True)

        # Temporal priors: pass previous-frame GT indices if enabled
        prev_sys = data.prev_gt_system_idx.to(device, non_blocking=True) if temporal_priors else None
        prev_bar = data.prev_gt_bar_page_idx.to(device, non_blocking=True) if temporal_priors else None

        # Only use scheduled sampling during training
        effective_p_pred = p_pred if train else 0.0

        autocast_context = (
            torch.autocast(device_type='cuda', enabled=True, dtype=amp_dtype)
            if amp and device.type == 'cuda'
            else nullcontext()
        )
        with torch.set_grad_enabled(train), autocast_context:
            model_outputs = network(
                score=scores, perf=perf,
                system_boxes=data.system_boxes,
                bar_boxes=data.bar_boxes,
                bars_per_system=data.bars_per_system,
                gt_system_idx=gt_system_idx,
                gt_bar_in_sys=gt_bar_in_sys,
                prev_gt_system_idx=prev_sys,
                prev_gt_bar_page_idx=prev_bar,
                tempo_aug=tempo_aug,
                p_pred=effective_p_pred,
            )

            # Use remapped bar targets if scheduled sampling is active
            bar_targets = model_outputs.get('bar_target_remapped', gt_bar_in_sys)
            ss_bar_valid = model_outputs.get('ss_bar_valid', None)

            # Combine gt_valid with scheduled sampling mask for bar/note
            if ss_bar_valid is not None:
                bar_note_valid = gt_valid & ss_bar_valid
            else:
                bar_note_valid = None

            loss_dict = criterion(
                model_outputs, gt_system_idx, bar_targets, gt_note_position,
                note_weight=note_weight, label_smoothing=label_smoothing,
                gt_valid=gt_valid, bar_note_valid=bar_note_valid,
            )
            loss = loss_dict['loss']

            if train:
                for boundary_name in ('z', 'audio_seq'):
                    boundary = model_outputs.get(boundary_name)
                    if isinstance(boundary, torch.Tensor) and boundary.requires_grad:
                        boundary.retain_grad()

            # Stack scalar metrics now, but defer their host transfer until
            # after backward. During training the optimizer guard copies them
            # together with the gradient norm in its one mandatory CUDA
            # synchronization. This avoids stalling once before backward and
            # a second time before the guarded optimizer step.
            tensor_keys = [
                key for key, value in loss_dict.items()
                if isinstance(value, torch.Tensor)
            ]
            numeric_keys = [
                key for key, value in loss_dict.items()
                if not isinstance(value, torch.Tensor)
                and isinstance(value, (int, float, np.number))
            ]
            metric_keys = tensor_keys + numeric_keys
            metric_tensors = [
                loss_dict[key].detach().float().reshape(()) for key in tensor_keys
            ]
            metric_tensors.extend(
                torch.as_tensor(
                    loss_dict[key], device=loss.device, dtype=torch.float32
                ).reshape(())
                for key in numeric_keys
            )
            metric_values = torch.stack(metric_tensors)

        if train:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()

            (
                stepped,
                grad_norm,
                bad_gradients,
                bad_value_count,
                recovered_norm_overflow,
                host_metric_values,
            ) = guarded_optimizer_step(
                network, criterion, optimizer, clip_grads, scaler=scaler,
                metric_values=metric_values,
            )
            if recovered_norm_overflow:
                recovered_grad_norm_overflows += 1
        else:
            host_metric_values = metric_values.cpu().tolist()

        resolved_values = dict(zip(metric_keys, host_metric_values))
        resolved_values.update({
            key: value for key, value in loss_dict.items()
            if not isinstance(value, (torch.Tensor, int, float, np.number))
        })

        # Never allow a non-finite batch to poison parameters, epoch aggregates,
        # or best-checkpoint selection. In training, guarded_optimizer_step has
        # already withheld the update when any stacked metric is non-finite.
        nonfinite_metrics = {
            key: value for key, value in resolved_values.items()
            if isinstance(value, (int, float, np.number))
            and not np.isfinite(value)
        }
        if nonfinite_metrics:
            output_nonfinite = {}
            output_nonfinite_rows = {}
            for key, value in model_outputs.items():
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    finite = torch.isfinite(value)
                    count = int((~finite).sum().item())
                    if count:
                        output_nonfinite[key] = count
                        # Batch-shaped tensors (not concatenated candidate
                        # logits) can identify the exact offending sample.
                        if value.ndim > 0 and value.shape[0] == len(data.file_names):
                            finite_rows = finite.reshape(value.shape[0], -1).all(dim=1)
                            output_nonfinite_rows[key] = torch.nonzero(
                                ~finite_rows, as_tuple=False
                            ).flatten().cpu().tolist()
            input_nonfinite = {
                'score': int((~torch.isfinite(data.scores)).sum().item()),
                'perf': [
                    int((~torch.isfinite(value)).sum().item())
                    for value in data.perf
                ],
            }
            input_stats = []
            for value in data.perf:
                fp32_value = value.detach().float()
                input_stats.append({
                    'samples': int(fp32_value.numel()),
                    'peak': float(fp32_value.abs().max().item())
                        if fp32_value.numel() else 0.0,
                    'rms': float(fp32_value.square().mean().sqrt().item())
                        if fp32_value.numel() else 0.0,
                })
            raise FloatingPointError(
                f"Non-finite batch {batch_idx} ({data.file_names}): "
                f"metrics={nonfinite_metrics}, "
                f"model_outputs={output_nonfinite}, "
                f"model_output_rows={output_nonfinite_rows}, "
                f"inputs={input_nonfinite}, input_stats={input_stats}"
            )

        for key, val in resolved_values.items():
            if key not in losses:
                losses[key] = AverageMeter()
            losses[key].update(val)

            if key not in recent_losses:
                recent_losses[key] = []
            recent_losses[key].append(val)
            if len(recent_losses[key]) > 10:
                recent_losses[key].pop(0)

        if train and not stepped:
            skipped_nonfinite_grad_batches += 1
            bad_preview = dict(list(bad_gradients.items())[:20])
            bad_tail = dict(list(bad_gradients.items())[-10:])
            bad_groups = {}
            for parameter_name, count in bad_gradients.items():
                parts = parameter_name.split('.')
                group_name = '.'.join(parts[:2]) if len(parts) > 1 else parts[0]
                group = bad_groups.setdefault(
                    group_name, {'tensors': 0, 'values': 0}
                )
                group['tensors'] += 1
                group['values'] += count
            boundary_gradients = {}
            for boundary_name in ('z', 'audio_seq'):
                boundary = model_outputs.get(boundary_name)
                gradient = boundary.grad if isinstance(boundary, torch.Tensor) else None
                if gradient is None:
                    boundary_gradients[boundary_name] = None
                    continue
                finite = torch.isfinite(gradient)
                finite_values = gradient[finite]
                boundary_gradients[boundary_name] = {
                    'shape': tuple(gradient.shape),
                    'nonfinite': int((~finite).sum().item()),
                    'max_finite_abs': (
                        float(finite_values.abs().max().item())
                        if finite_values.numel() else None
                    ),
                }
            warning = (
                f'[gradient guard] skipped batch {batch_idx} '
                f'({data.file_names}): total_norm={grad_norm}, '
                f'nonfinite_values={bad_value_count}, '
                f'parameter_groups={bad_groups}, '
                f'parameter_tensors={len(bad_gradients)}, '
                f'first_parameters={bad_preview}, '
                f'last_parameters={bad_tail}, '
                f'boundary_gradients={boundary_gradients}'
            )
            if is_main_process():
                print(f'\nWARNING: {warning}')
            if skipped_nonfinite_grad_batches > max_nonfinite_grad_skips:
                raise FloatingPointError(
                    f'Exceeded max_nonfinite_grad_skips='
                    f'{max_nonfinite_grad_skips}; latest {warning}'
                )

        if is_main_process():
            current_loss = np.mean(recent_losses.get('loss', [0]))
            sys_loss = np.mean(recent_losses.get('sys_loss', [0]))
            bar_loss = np.mean(recent_losses.get('bar_loss', [0]))
            note_loss = np.mean(recent_losses.get('note_loss', [0]))
            sys_acc = np.mean(recent_losses.get('sys_acc', [0]))
            bar_acc = np.mean(recent_losses.get('bar_acc', [0]))

            progress_bar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'sys': f'{sys_loss:.4f}',
                'bar': f'{bar_loss:.4f}',
                'note': f'{note_loss:.4f}',
                'avg': f'{losses["loss"].avg:.4f}',
                'sAcc': f'{sys_acc:.2f}',
                'bAcc': f'{bar_acc:.2f}',
            }, refresh=False)
            progress_bar.update(1)

    stats = {}
    for key in losses:
        stats[key] = losses[key].avg
    stats['skipped_nonfinite_grad_batches'] = skipped_nonfinite_grad_batches
    stats['recovered_grad_norm_overflows'] = recovered_grad_norm_overflows

    if is_main_process():
        progress_bar.close()

    return stats


def train(args):
    # Keep the batch-loop helpers importable in lightweight CPU test and
    # profiling environments; model construction pulls optional audio/CUDA
    # dependencies that are only needed by the full training entry point.
    from coda.models.coda_model import build_model

    init_distributed_mode(args)
    device = torch.device(args.device)

    amp_dtype = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }[args.amp_dtype]
    if (
        args.amp and device.type == 'cuda'
        and amp_dtype == torch.bfloat16
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError(
            '--amp_dtype bfloat16 requires a CUDA GPU with BF16 support; '
            'use --amp_dtype float16 on this device'
        )

    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    resume_checkpoint = None
    if args.resume_state is not None:
        print(f'Loading resume state from {args.resume_state}')
        resume_checkpoint = torch.load(args.resume_state, map_location='cpu')
        validate_resume_checkpoint(resume_checkpoint, args.resume_state)

    # Keep a resumed phase on its original deterministic data stream even if
    # the caller accidentally changes --tag, while giving Phase 1 and Phase 2
    # distinct permutations and augmentations under their normal distinct tags.
    checkpoint_args = (
        resume_checkpoint.get('args', {}) if resume_checkpoint is not None else {}
    )
    sampling_tag = checkpoint_args.get('tag', args.tag)
    training_data_seed = training_phase_seed(seed, sampling_tag)

    dump_path = None
    logging = not args.no_log

    if is_main_process() and logging:
        if resume_checkpoint is not None:
            dump_path = os.path.dirname(os.path.abspath(args.resume_state))
            time_stamp = os.path.basename(dump_path) + "_resume"
        else:
            time_stamp = strftime("%Y%m%d_%H%M%S", gmtime()) + f"_{args.tag}"

        if not os.path.exists(args.log_root):
            os.makedirs(args.log_root)
        if not os.path.exists(args.dump_root):
            os.makedirs(args.dump_root)

        if dump_path is None:
            dump_path = os.path.join(args.dump_root, time_stamp)
        os.makedirs(dump_path, exist_ok=True)

    total_epochs = args.num_epochs

    log_writer = None
    if is_main_process() and logging:
        log_dir = os.path.join(args.log_root, time_stamp)
        log_writer = SummaryWriter(log_dir=log_dir)

        text = ""
        for arg in sorted(vars(args)):
            text += f"**{arg}:** {getattr(args, arg)}<br>"
        log_writer.add_text("run_config", text)
        log_writer.add_text("cmd", " ".join(sys.argv))

        with open(os.path.join(dump_path, 'net_config.json'), "w") as f:
            json.dump(args.config, f)

    # Initialize wandb
    wandb_run = None
    if is_main_process() and args.wandb and WANDB_AVAILABLE:
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.tag or time_stamp,
            config={**args.config, 'batch_size': args.batch_size, 'lr': args.learning_rate,
                    'note_weight': args.note_weight},
            tags=['selection'],
            reinit=True,
        )

    # Build model + criterion
    network, criterion = build_model(
        args.config,
        loss_calibration=args.loss_calibration,
        label_smoothing=args.label_smoothing,
    )

    initialization_checkpoint = resume_checkpoint
    if initialization_checkpoint is None and args.param_path is not None:
        print(f'Loading model from {args.param_path}')
        initialization_checkpoint = torch.load(args.param_path, map_location='cpu')

    if initialization_checkpoint is not None:
        state_dict = extract_model_state(initialization_checkpoint)
        network.load_state_dict(state_dict, strict=True)

        criterion_state = initialization_checkpoint.get('criterion_state_dict')
        if isinstance(criterion, nn.Module) and criterion_state is not None:
            criterion.load_state_dict(criterion_state)

    network.to(device)

    # If uncertainty weighting, criterion has learnable params -- move to device
    if isinstance(criterion, nn.Module):
        criterion.to(device)

    print(f"Model parameters: {sum(p.numel() for p in network.parameters() if p.requires_grad):,}")
    if isinstance(criterion, nn.Module):
        print(f"Loss parameters: {sum(p.numel() for p in criterion.parameters() if p.requires_grad):,} "
              f"(uncertainty weighting)")

    # Dataset (predict_sb=True for targets that include system/bar boxes)
    train_dataset = load_dataset(
        args.train_sets, augment=args.augment, scale_width=args.scale_width,
        split_files=args.train_split_files, ir_path=args.ir_path,
        load_audio=args.load_audio, predict_sb=True,
        cold_start_prob=args.cold_start_prob,
        jump_prob=args.jump_prob,
    )
    val_dataset = load_dataset(
        args.val_sets, augment=False, scale_width=args.scale_width,
        split_files=args.val_split_files, load_audio=args.load_audio,
        predict_sb=True,
    )

    # Wrap with selection metadata
    train_sel = SelectionDatasetWrapper(
        train_dataset, augmentation_seed=training_data_seed
    )
    val_sel = SelectionDatasetWrapper(val_dataset)

    batch_size = args.batch_size
    sampler_train = EpochSeededRandomSampler(train_sel, seed=training_data_seed)
    sampler_val = SequentialSampler(val_sel)
    batch_sampler_train = BatchSampler(
        sampler_train, batch_size, drop_last=True
    )

    train_num_workers = args.num_workers
    val_num_workers = (
        train_num_workers
        if args.val_num_workers is None
        else args.val_num_workers
    )
    train_use_persistent = train_num_workers > 0
    val_use_persistent = val_num_workers > 0
    print(
        "Data-loader workers: "
        f"train={train_num_workers}, validation={val_num_workers}"
    )
    # DataLoader iterator creation draws a worker base seed. Keep that draw on
    # private generators so a process restart cannot advance the model's
    # restored CPU RNG (which drives dropout and other training stochasticity).
    train_loader_generator = torch.Generator().manual_seed(
        stable_named_seed(training_data_seed, "training-loader-workers")
    )
    val_loader_generator = torch.Generator().manual_seed(
        stable_named_seed(training_data_seed, "validation-loader-workers")
    )
    train_loader = DataLoader(
        train_sel, batch_sampler=batch_sampler_train,
        collate_fn=selection_collate_wrapper, num_workers=train_num_workers,
        pin_memory=device.type == 'cuda',
        persistent_workers=train_use_persistent,
        prefetch_factor=2 if train_use_persistent else None,
        generator=train_loader_generator,
    )
    val_loader = DataLoader(
        val_sel, batch_size, sampler=sampler_val, drop_last=False,
        collate_fn=selection_collate_wrapper, num_workers=val_num_workers,
        pin_memory=device.type == 'cuda',
        persistent_workers=val_use_persistent,
        prefetch_factor=2 if val_use_persistent else None,
        generator=val_loader_generator,
    )

    # Optimizer setup. Coverage is checked explicitly so direct parameters
    # such as Mamba A/D, MultiheadAttention input projections, layer scales,
    # and transition logits cannot silently be omitted.
    extra_modules = [criterion] if isinstance(criterion, nn.Module) else None
    optim, optim_summary = build_adamw(
        network,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        extra_modules=extra_modules,
    )
    print(
        "Optimizer coverage: "
        f"{optim_summary['total_parameters']:,} parameters "
        f"({optim_summary['decay_parameters']:,} decay, "
        f"{optim_summary['no_decay_parameters']:,} no-decay)"
    )

    lrf = args.learning_rate_factor
    lf = lambda x: ((1 + math.cos(x * math.pi / total_epochs)) / 2) * (1 - lrf) + lrf
    scheduler = LambdaLR(optim, lr_lambda=lf)
    # BF16 shares FP32's exponent range and does not require loss scaling.
    scaler = torch.cuda.amp.GradScaler(
        enabled=(
            args.amp and device.type == 'cuda'
            and amp_dtype == torch.float16
        )
    )

    min_loss = np.inf
    best_streaming_bar_acc = -1.0
    best_epoch = -1
    start_epoch = 0

    if resume_checkpoint is not None:
        optim.load_state_dict(resume_checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
        if scaler.is_enabled() and resume_checkpoint.get('scaler_state_dict') is not None:
            scaler.load_state_dict(resume_checkpoint['scaler_state_dict'])
        min_loss = float(resume_checkpoint.get('min_loss', min_loss))
        best_streaming_bar_acc = float(
            resume_checkpoint.get('best_streaming_bar_acc', best_streaming_bar_acc)
        )
        best_epoch = int(resume_checkpoint.get('best_epoch', best_epoch))
        start_epoch = int(resume_checkpoint['epoch']) + 1
        restore_rng_state(resume_checkpoint)
        print(f'Resuming at epoch {start_epoch} of {total_epochs}')

    # Discover streaming eval pieces from val directory
    streaming_eval_dir = args.streaming_eval_dir or (args.val_sets[0] if args.val_sets else None)
    streaming_pieces_all = []
    streaming_pieces_tier1 = []
    if streaming_eval_dir and os.path.isdir(streaming_eval_dir):
        import glob as glob_mod
        npz_files = sorted(glob_mod.glob(os.path.join(streaming_eval_dir, '*.npz')))
        streaming_pieces_all = [os.path.basename(f)[:-4] for f in npz_files]

        if args.streaming_eval_pieces:
            # User specified explicit tier-1 pieces
            streaming_pieces_tier1 = args.streaming_eval_pieces
        else:
            # Auto-select tier-1 subset: first N pieces (sorted by name for reproducibility)
            n_tier1 = min(args.streaming_eval_n, len(streaming_pieces_all))
            streaming_pieces_tier1 = streaming_pieces_all[:n_tier1]

    use_streaming_selection = len(streaming_pieces_tier1) > 0
    streaming_eval_interval = args.streaming_eval_full_interval

    print(f"\n{'='*60}")
    print(f"[Selection Training Config]")
    print(f"  Epochs: {total_epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {args.learning_rate} (cosine -> {args.learning_rate * lrf:.6f})")
    print(f"  Note Weight: {args.note_weight} ({'ignored' if args.loss_calibration == 'uncertainty' else 'active'})")
    print(f"  Loss Calibration: {args.loss_calibration}")
    print(f"  Label Smoothing: {args.label_smoothing}")
    print(f"  Cold-start Prob: {args.cold_start_prob}")
    print(f"  Jump Prob: {args.jump_prob}")
    print(f"  Deterministic data seed: {training_data_seed} ({sampling_tag})")
    print(f"  Weight Decay: {args.weight_decay}")
    print(f"  Grad Clip: {args.clip_grads}")
    amp_description = args.amp_dtype if args.amp and device.type == 'cuda' else 'OFF'
    print(f"  AMP: {amp_description}")
    print(f"  Augmentation: {'ON' if args.augment else 'OFF'}")
    print(f"  Temporal Priors in Training: {'ON' if args.temporal_priors else 'OFF'}")
    if args.scheduled_sampling:
        print(f"  Scheduled Sampling: ON (p_pred 0 -> {args.ss_max_p} over {args.ss_ramp_epochs} epochs)")
    else:
        print(f"  GT routing always used for bar/note stages")
    if use_streaming_selection:
        print(f"  Streaming eval dir: {streaming_eval_dir}")
        print(f"  Tier-1 pieces ({len(streaming_pieces_tier1)}): {streaming_pieces_tier1[:5]}{'...' if len(streaming_pieces_tier1) > 5 else ''}")
        print(f"  Tier-2 pieces ({len(streaming_pieces_all)}): every {streaming_eval_interval} epochs")
        print(f"  Model selection: streaming bar accuracy (tier-1)")
    else:
        print(f"  Model selection: teacher-forced val loss (no streaming eval)")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, total_epochs):
        sampler_train.set_epoch(epoch)
        # Compute scheduled sampling probability for this epoch
        if args.scheduled_sampling:
            progress = min(1.0, epoch / max(args.ss_ramp_epochs - 1, 1))
            p_pred = progress * args.ss_max_p
        else:
            p_pred = 0.0

        if is_main_process():
            print(f"\n--- Epoch {epoch} ---")
            if p_pred > 0:
                print(f"  [scheduled sampling] p_pred={p_pred:.3f}")

        network.train()
        try:
            tr_stats = iterate_selection(
                network, train_loader, criterion, optimizer=optim,
                clip_grads=args.clip_grads, device=device, tempo_aug=args.augment,
                note_weight=args.note_weight, label_smoothing=args.label_smoothing,
                epoch=epoch, temporal_priors=args.temporal_priors,
                p_pred=p_pred,
                amp=args.amp, amp_dtype=amp_dtype, scaler=scaler,
                max_batches=args.max_train_batches,
                max_nonfinite_grad_skips=args.max_nonfinite_grad_skips,
            )
        except Exception as exc:
            # Record the failed epoch for diagnosis. Training resumes only
            # from checkpoints written at completed epoch boundaries.
            if is_main_process() and logging:
                failure_checkpoint = make_training_checkpoint(
                    network, criterion, optim, scheduler, scaler, epoch - 1,
                    min_loss=min_loss,
                    best_streaming_bar_acc=best_streaming_bar_acc,
                    best_epoch=best_epoch,
                    args=vars(args),
                )
                failure_checkpoint.update({
                    'partial_epoch': True,
                    'resumable': False,
                    'failure_epoch': int(epoch),
                    'failure_type': type(exc).__name__,
                    'failure_message': str(exc),
                    'resume_from_completed_checkpoint': os.path.join(
                        dump_path, 'latest_checkpoint.pt'
                    ),
                })
                failure_path = os.path.join(dump_path, 'failure_checkpoint.pt')
                atomic_torch_save(failure_checkpoint, failure_path)
                print(f'Diagnostic failure checkpoint: {failure_path}')
                print(
                    'The failure checkpoint is diagnostic only; resume '
                    f"from {os.path.join(dump_path, 'latest_checkpoint.pt')}"
                )
            raise

        network.eval()
        val_stats = iterate_selection(
            network, val_loader, criterion, optimizer=None,
            device=device, note_weight=args.note_weight,
            label_smoothing=args.label_smoothing, epoch=epoch,
            temporal_priors=args.temporal_priors,
            amp=args.amp, amp_dtype=amp_dtype,
            max_batches=args.max_val_batches,
        )

        tr_stats = {k: torch.FloatTensor([v]).to(device) for k, v in tr_stats.items() if isinstance(v, float)}
        val_stats = {k: torch.FloatTensor([v]).to(device) for k, v in val_stats.items() if isinstance(v, float)}
        tr_stats = reduce_dict(tr_stats, average=True)
        val_stats = reduce_dict(val_stats, average=True)

        tr_loss = tr_stats['loss'].item()
        val_loss = val_stats['loss'].item()

        # --- Streaming evaluation ---
        streaming_metrics = None
        if is_main_process() and use_streaming_selection:
            # Tier 1: every epoch on subset
            print(f"\n  [Streaming Eval] Tier-1 ({len(streaming_pieces_tier1)} pieces)...")
            streaming_metrics = streaming_eval(
                network, streaming_eval_dir, streaming_pieces_tier1,
                args.scale_width, device, verbose=True
            )

            # Tier 2: full eval every N epochs
            if (epoch + 1) % streaming_eval_interval == 0 and len(streaming_pieces_all) > len(streaming_pieces_tier1):
                print(f"\n  [Streaming Eval] Tier-2 FULL ({len(streaming_pieces_all)} pieces)...")
                full_metrics = streaming_eval(
                    network, streaming_eval_dir, streaming_pieces_all,
                    args.scale_width, device, verbose=True
                )
                # Log tier-2 metrics separately
                if log_writer:
                    log_writer.add_scalar('streaming_full/sys_acc', full_metrics['sys_acc'], epoch)
                    log_writer.add_scalar('streaming_full/bar_acc', full_metrics['bar_acc'], epoch)
                    log_writer.add_scalar('streaming_full/mean_px_err', full_metrics['mean_px_err'], epoch)
                if wandb_run is not None:
                    wandb.log({
                        'streaming_full/sys_acc': full_metrics['sys_acc'],
                        'streaming_full/bar_acc': full_metrics['bar_acc'],
                        'streaming_full/mean_px_err': full_metrics['mean_px_err'],
                        'epoch': epoch,
                    })

        # --- Model selection ---
        if use_streaming_selection and streaming_metrics is not None:
            # Select by streaming bar accuracy
            current_bar_acc = streaming_metrics['bar_acc']
            is_best = current_bar_acc > best_streaming_bar_acc
            if is_best:
                best_streaming_bar_acc = current_bar_acc
        else:
            # Select by teacher-forced validation loss when streaming
            # selection is not scheduled for this epoch.
            is_best = val_loss < min_loss

        if is_best:
            min_loss = min(min_loss, val_loss)
            best_epoch = epoch
            color = '\033[92m'
            if is_main_process() and logging:
                print("Store best model...")
                atomic_torch_save(network.state_dict(), os.path.join(dump_path, "best_model.pt"))
        else:
            color = '\033[91m'

        if is_main_process() and logging:
            atomic_torch_save(network.state_dict(), os.path.join(dump_path, "latest_model.pt"))

        if is_main_process() and logging and log_writer:
            log_writer.add_scalar('training/lr', optim.param_groups[0]['lr'], epoch)

            for key in tr_stats:
                if 'loss' in key or 'acc' in key:
                    log_writer.add_scalar(f'training/{key}', tr_stats[key].item(), epoch)
                    log_writer.add_scalar(f'validation/{key}', val_stats[key].item(), epoch)

            if streaming_metrics is not None:
                log_writer.add_scalar('streaming/sys_acc', streaming_metrics['sys_acc'], epoch)
                log_writer.add_scalar('streaming/bar_acc', streaming_metrics['bar_acc'], epoch)
                log_writer.add_scalar('streaming/mean_px_err', streaming_metrics['mean_px_err'], epoch)

        if wandb_run is not None:
            wandb_log = {
                'epoch': epoch,
                'train/loss': tr_loss, 'val/loss': val_loss,
                'lr': optim.param_groups[0]['lr'],
            }
            for key in tr_stats:
                if key != 'loss':
                    wandb_log[f'train/{key}'] = tr_stats[key].item()
                    wandb_log[f'val/{key}'] = val_stats[key].item()
            if streaming_metrics is not None:
                wandb_log['streaming/sys_acc'] = streaming_metrics['sys_acc']
                wandb_log['streaming/bar_acc'] = streaming_metrics['bar_acc']
                wandb_log['streaming/mean_px_err'] = streaming_metrics['mean_px_err']
            wandb.log(wandb_log)
            if is_best:
                if streaming_metrics is not None:
                    wandb.run.summary['best_streaming_bar_acc'] = best_streaming_bar_acc
                wandb.run.summary['best_val_loss'] = val_loss
                wandb.run.summary['best_epoch'] = epoch

        val_sys = val_stats.get('sys_loss', torch.tensor(0)).item()
        val_bar = val_stats.get('bar_loss', torch.tensor(0)).item()
        val_note = val_stats.get('note_loss', torch.tensor(0)).item()
        val_sacc = val_stats.get('sys_acc', torch.tensor(0)).item()
        val_bacc = val_stats.get('bar_acc', torch.tensor(0)).item()

        print(f"{color}Epoch {epoch} | Train: {tr_loss:.6f} | Val: {val_loss:.6f}\033[0m")
        print(f"  sys_loss={val_sys:.4f} bar_loss={val_bar:.4f} note_loss={val_note:.4f} "
              f"sys_acc={val_sacc:.2f} bar_acc={val_bacc:.2f}")
        if 'w_sys' in tr_stats:
            w_s = tr_stats.get('w_sys', torch.tensor(0)).item() if isinstance(tr_stats.get('w_sys'), torch.Tensor) else tr_stats.get('w_sys', 0)
            w_b = tr_stats.get('w_bar', torch.tensor(0)).item() if isinstance(tr_stats.get('w_bar'), torch.Tensor) else tr_stats.get('w_bar', 0)
            w_n = tr_stats.get('w_note', torch.tensor(0)).item() if isinstance(tr_stats.get('w_note'), torch.Tensor) else tr_stats.get('w_note', 0)
            print(f"  [uncertainty] w_sys={w_s:.3f} w_bar={w_b:.3f} w_note={w_n:.3f}")
        if streaming_metrics is not None:
            print(f"  [streaming] sys={streaming_metrics['sys_acc']:.3f} "
                  f"bar={streaming_metrics['bar_acc']:.3f} "
                  f"px_err={streaming_metrics['mean_px_err']:.1f} "
                  f"{'*** BEST ***' if is_best else ''}")

        scheduler.step()

        if is_main_process() and logging:
            checkpoint_metrics = {
                'train_loss': float(tr_loss),
                'val_loss': float(val_loss),
                'train': {
                    key: float(value.item() if isinstance(value, torch.Tensor) else value)
                    for key, value in tr_stats.items()
                },
                'validation': {
                    key: float(value.item() if isinstance(value, torch.Tensor) else value)
                    for key, value in val_stats.items()
                },
                'streaming': {
                    key: float(value)
                    for key, value in (streaming_metrics or {}).items()
                    if isinstance(value, (int, float, np.number))
                },
            }
            checkpoint = make_training_checkpoint(
                network, criterion, optim, scheduler, scaler, epoch,
                min_loss=min_loss,
                best_streaming_bar_acc=best_streaming_bar_acc,
                best_epoch=best_epoch,
                args=vars(args),
                metrics=checkpoint_metrics,
            )
            atomic_torch_save(checkpoint, os.path.join(dump_path, "latest_checkpoint.pt"))
            if is_best:
                atomic_torch_save(checkpoint, os.path.join(dump_path, "best_checkpoint.pt"))
            if args.save_every_epoch:
                epoch_path = os.path.join(
                    dump_path, f"checkpoint_epoch_{epoch:03d}.pt"
                )
                atomic_torch_save(checkpoint, epoch_path)

    if wandb_run is not None:
        wandb.finish()


if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description='Train Selection Model')
    parser.add_argument('--augment', default=False, action='store_true')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dump_root', type=str, default='params')
    parser.add_argument('--ir_path', type=str, default=None, nargs='+')
    parser.add_argument('--load_audio', default=False, action='store_true')
    parser.add_argument('--log_root', type=str, default='runs')
    parser.add_argument('--no_log', default=False, action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument(
        '--val_num_workers', default=None, type=int,
        help='Validation loader workers (default: match --num_workers)',
    )
    parser.add_argument('--max_train_batches', default=None, type=int,
                        help=argparse.SUPPRESS)
    parser.add_argument('--max_val_batches', default=None, type=int,
                        help=argparse.SUPPRESS)
    parser.add_argument('--param_path', type=str, default=None)
    parser.add_argument('--resume_state', type=str, default=None,
                        help='Resume the same run from a structured *_checkpoint.pt file')
    parser.add_argument('--seed', type=int, default=4711)
    parser.add_argument('--tag', type=str, required=True)
    parser.add_argument('--scale_width', type=int, default=416)
    parser.add_argument('--train_sets', nargs='+', required=True)
    parser.add_argument('--train_split_files', default=None, nargs='+')
    parser.add_argument('--val_sets', nargs='+', required=True)
    parser.add_argument('--val_split_files', default=None, nargs='+')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--clip_grads', type=float, default=1.0)
    parser.add_argument('--max_nonfinite_grad_skips', type=int, default=8,
                        help='Maximum protected non-finite-gradient skips per epoch')
    parser.add_argument('--learning_rate', '--lr', type=float, default=5e-4)
    parser.add_argument('--learning_rate_factor', '--lrf', type=float, default=0.01)
    parser.add_argument('--note_weight', type=float, default=10.0,
                        help='Weight for note MSE loss (default: 10.0 to balance vs CE losses)')
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--amp', default=False, action='store_true',
                        help='Use CUDA automatic mixed precision for faster training')
    parser.add_argument('--amp_dtype', default='float16',
                        choices=['float16', 'bfloat16'],
                        help='CUDA autocast dtype (BF16 is recommended on supported GPUs)')

    # Loss calibration and regularization
    parser.add_argument('--loss_calibration', type=str, default='static',
                        choices=['static', 'uncertainty'],
                        help='Loss weighting: static (manual weights) or uncertainty (learned)')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing for system/bar CE losses (default: 0.0)')
    parser.add_argument('--cold_start_prob', type=float, default=0.0,
                        help='Cold-start regularization probability (default: 0.0)')
    parser.add_argument('--jump_prob', type=float, default=0.0,
                        help='On-the-fly jump augmentation probability per sample (e.g., 0.1). '
                             'Swaps destination while keeping biased temporal priors.')
    parser.add_argument('--temporal_priors', default=False, action='store_true',
                        help='Add learnable temporal priors to logits during training (Phase 2a)')

    # Scheduled sampling (Phase 2)
    parser.add_argument('--scheduled_sampling', default=False, action='store_true',
                        help='Enable scheduled sampling: gradually use predicted routing')
    parser.add_argument('--ss_max_p', type=float, default=0.7,
                        help='Max probability of predicted routing (default: 0.7)')
    parser.add_argument('--ss_ramp_epochs', type=int, default=5,
                        help='Epochs to ramp p_pred from 0 to ss_max_p (default: 5)')

    # Streaming evaluation (model selection by online accuracy)
    parser.add_argument('--streaming_eval_dir', type=str, default=None,
                        help='Directory with val pieces for streaming eval (default: first val_set)')
    parser.add_argument('--streaming_eval_pieces', nargs='+', default=None,
                        help='Explicit tier-1 piece names for streaming eval')
    parser.add_argument('--streaming_eval_n', type=int, default=5,
                        help='Number of tier-1 pieces if not explicitly specified (default: 5)')
    parser.add_argument('--streaming_eval_full_interval', type=int, default=3,
                        help='Run full tier-2 streaming eval every N epochs (default: 3)')

    parser.add_argument('--save_every_epoch', default=False, action='store_true',
                        help='Save a full resumable checkpoint for every epoch '
                             '(checkpoint_epoch_000.pt, checkpoint_epoch_001.pt, ...)')

    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--wandb_project', type=str, default='score-follower')
    parser.add_argument('--wandb_entity', type=str, default=None)

    parser.add_argument('--dist_url', default='env://')

    args = parser.parse_args()
    if args.num_workers < 0:
        parser.error('--num_workers must be non-negative')
    if args.val_num_workers is not None and args.val_num_workers < 0:
        parser.error('--val_num_workers must be non-negative')
    args.config = load_yaml(args.config)
    train(args)
