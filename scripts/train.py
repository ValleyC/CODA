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
        --param_path params/<phase1_dir>/best_model.pt \
        --tag selection_v2_phase2 --temporal_priors --augment \
        --scheduled_sampling --ss_max_p 0.7 --ss_ramp_epochs 5
"""

import argparse
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

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, BatchSampler
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from time import gmtime, strftime

from coda.utils.dist_utils import (
    init_distributed_mode, is_main_process, get_rank, reduce_dict
)
from coda.utils.general import load_yaml, AverageMeter
from coda.dataset import (
    load_dataset, selection_collate_wrapper, selection_getitem
)
from coda.models.coda_model import build_model
from coda.utils.loss import selection_loss
from coda.utils.streaming_eval import streaming_eval


class SelectionDatasetWrapper(Dataset):
    """Wraps SequenceDataset to add page layout metadata via selection_getitem."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return selection_getitem(self.dataset, item)


def iterate_selection(network, dataloader, criterion, optimizer=None,
                      clip_grads=None, device=torch.device('cuda'),
                      tempo_aug=False, note_weight=1.0, label_smoothing=0.0,
                      epoch=0, temporal_priors=False, p_pred=0.0):
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

    mode_str = "Train" if train else "Val"
    if is_main_process():
        progress_bar = tqdm(total=len(dataloader), ncols=140, desc=f"{mode_str} E{epoch}")

    for batch_idx, data in enumerate(dataloader):
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

        with torch.set_grad_enabled(train):
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

            for key in loss_dict:
                if key not in losses:
                    losses[key] = AverageMeter()
                val = loss_dict[key].item() if isinstance(loss_dict[key], torch.Tensor) else loss_dict[key]
                losses[key].update(val)

                if key not in recent_losses:
                    recent_losses[key] = []
                recent_losses[key].append(val)
                if len(recent_losses[key]) > 10:
                    recent_losses[key].pop(0)

        if train:
            optimizer.zero_grad()
            loss.backward()

            # Clip gradients on ALL parameters
            if clip_grads is not None:
                net = network if hasattr(network, 'conditioning_network') else network.module
                clip_grad_norm_(net.parameters(), clip_grads)

            optimizer.step()

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
            })
            progress_bar.update(1)

    stats = {}
    for key in losses:
        stats[key] = losses[key].avg

    if is_main_process():
        progress_bar.close()

    return stats


def train(args):
    init_distributed_mode(args)
    device = torch.device(args.device)

    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True

    dump_path = None
    logging = not args.no_log

    if is_main_process() and logging:
        time_stamp = strftime("%Y%m%d_%H%M%S", gmtime()) + f"_{args.tag}"

        if not os.path.exists(args.log_root):
            os.makedirs(args.log_root)
        if not os.path.exists(args.dump_root):
            os.makedirs(args.dump_root)

        dump_path = os.path.join(args.dump_root, time_stamp)
        if not os.path.exists(dump_path):
            os.mkdir(dump_path)

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

    if args.param_path is not None:
        print(f'Loading model from {args.param_path}')
        state_dict = torch.load(args.param_path, map_location='cpu')
        # Handle DataParallel-wrapped checkpoints
        if any(k.startswith('module.') for k in state_dict):
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        try:
            network.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            missing, unexpected = network.load_state_dict(state_dict, strict=False)
            # Only allow new bar_head cross-attention params to be missing
            allowed_prefixes = ('bar_head.audio_proj.', 'bar_head.cross_attn.',
                                'bar_head.attn_ln.', 'bar_head.attn_out_proj.')
            allowed_exact = ('bar_head.layer_scale',)
            for key in missing:
                ok = any(key.startswith(p) for p in allowed_prefixes) or key in allowed_exact
                assert ok, f"Unexpected missing key: {key}"
            assert not unexpected, f"Unexpected keys in checkpoint: {unexpected}"
            print(f"  Partial load: {len(missing)} new bar cross-attn params initialized randomly")

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
    train_sel = SelectionDatasetWrapper(train_dataset)
    val_sel = SelectionDatasetWrapper(val_dataset)

    batch_size = args.batch_size
    sampler_train = RandomSampler(train_sel)
    sampler_val = SequentialSampler(val_sel)
    batch_sampler_train = BatchSampler(sampler_train, batch_size, drop_last=True)

    use_persistent = args.num_workers > 0
    train_loader = DataLoader(
        train_sel, batch_sampler=batch_sampler_train,
        collate_fn=selection_collate_wrapper, num_workers=args.num_workers,
        pin_memory=False, persistent_workers=use_persistent,
        prefetch_factor=2 if use_persistent else None,
    )
    val_loader = DataLoader(
        val_sel, batch_size, sampler=sampler_val, drop_last=False,
        collate_fn=selection_collate_wrapper, num_workers=args.num_workers,
        pin_memory=False, persistent_workers=use_persistent,
        prefetch_factor=2 if use_persistent else None,
    )

    # Optimizer setup
    pg0, pg1, pg2 = [], [], []
    for k, v in network.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            pg0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)

    optim = torch.optim.AdamW(pg0, lr=args.learning_rate)
    optim.add_param_group({'params': pg1, 'weight_decay': args.weight_decay})
    optim.add_param_group({'params': pg2})

    # Add criterion params (uncertainty weighting log-variances) if applicable
    if isinstance(criterion, nn.Module):
        criterion_params = list(criterion.parameters())
        if criterion_params:
            optim.add_param_group({'params': criterion_params, 'lr': args.learning_rate})

    lrf = args.learning_rate_factor
    lf = lambda x: ((1 + math.cos(x * math.pi / total_epochs)) / 2) * (1 - lrf) + lrf
    scheduler = LambdaLR(optim, lr_lambda=lf)

    min_loss = np.inf
    best_streaming_bar_acc = -1.0

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
    print(f"  Weight Decay: {args.weight_decay}")
    print(f"  Grad Clip: {args.clip_grads}")
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

    for epoch in range(total_epochs):
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
        tr_stats = iterate_selection(
            network, train_loader, criterion, optimizer=optim,
            clip_grads=args.clip_grads, device=device, tempo_aug=args.augment,
            note_weight=args.note_weight, label_smoothing=args.label_smoothing,
            epoch=epoch, temporal_priors=args.temporal_priors,
            p_pred=p_pred,
        )

        network.eval()
        val_stats = iterate_selection(
            network, val_loader, criterion, optimizer=None,
            device=device, note_weight=args.note_weight,
            label_smoothing=args.label_smoothing, epoch=epoch,
            temporal_priors=args.temporal_priors,
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
            # Fallback: teacher-forced val loss
            is_best = val_loss < min_loss

        if is_best:
            min_loss = min(min_loss, val_loss)
            color = '\033[92m'
            if is_main_process() and logging:
                print("Store best model...")
                torch.save(network.state_dict(), os.path.join(dump_path, "best_model.pt"))
        else:
            color = '\033[91m'
            if is_main_process() and logging:
                torch.save(network.state_dict(), os.path.join(dump_path, "latest_model.pt"))

        # Save per-epoch checkpoint
        if is_main_process() and logging and args.save_every_epoch:
            epoch_path = os.path.join(dump_path, f"epoch_{epoch}.pt")
            torch.save(network.state_dict(), epoch_path)

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
    parser.add_argument('--param_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=4711)
    parser.add_argument('--tag', type=str, required=True)
    parser.add_argument('--scale_width', type=int, default=416)
    parser.add_argument('--train_sets', nargs='+', required=True)
    parser.add_argument('--train_split_files', default=None, nargs='+')
    parser.add_argument('--val_sets', nargs='+', required=True)
    parser.add_argument('--val_split_files', default=None, nargs='+')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--clip_grads', type=float, default=1.0)
    parser.add_argument('--learning_rate', '--lr', type=float, default=5e-4)
    parser.add_argument('--learning_rate_factor', '--lrf', type=float, default=0.01)
    parser.add_argument('--note_weight', type=float, default=10.0,
                        help='Weight for note MSE loss (default: 10.0 to balance vs CE losses)')
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--weight_decay', type=float, default=0.0005)

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
                        help='Save checkpoint for every epoch (epoch_0.pt, epoch_1.pt, ...)')

    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--wandb_project', type=str, default='score-follower')
    parser.add_argument('--wandb_entity', type=str, default=None)

    parser.add_argument('--dist_url', default='env://')

    args = parser.parse_args()
    args.config = load_yaml(args.config)
    train(args)
