"""Optimizer construction with explicit parameter-coverage guarantees."""

import math

from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn


def clip_gradient_at_boundary(gradient: torch.Tensor, max_norm: float) -> torch.Tensor:
    """Sanitize and norm-clip an activation gradient without FP32 overflow.

    Module-boundary hooks run before parameter gradients exist, so they can
    prevent an exploding downstream Jacobian from overflowing an upstream
    recurrent backward pass. The max-rescaled norm avoids squaring raw FP32
    values that may be finite but close to the datatype's dynamic limit.
    """
    clean = torch.nan_to_num(gradient)
    max_abs = clean.detach().abs().max()
    safe_scale = torch.where(max_abs > 0, max_abs, torch.ones_like(max_abs))
    normalized = clean / safe_scale.to(dtype=clean.dtype)
    normalized_norm = torch.linalg.vector_norm(normalized.float(), 2)
    total_norm = safe_scale.double() * normalized_norm.double()
    coefficient = torch.clamp(max_norm / (total_norm + 1e-12), max=1.0)
    return clean * coefficient.to(device=clean.device, dtype=clean.dtype)


def guarded_optimizer_step(
    network: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    clip_grads: Optional[float],
    scaler=None,
    metric_values: Optional[torch.Tensor] = None,
):
    """Apply an optimizer step only when every gradient is finite.

    ``clip_grad_norm_`` returns the pre-clipping total norm. Checking that
    value before ``optimizer.step`` closes an important BF16 safety gap:
    unlike FP16 GradScaler, BF16 autocast has no automatic skipped-step
    mechanism. A finite forward pass can therefore produce an overflowing
    backward pass and poison every parameter on the following update.

    ``metric_values`` optionally supplies detached scalar training metrics.
    They are copied to the host together with the gradient norm, so the
    ordinary training path pays one CUDA synchronization rather than a
    separate synchronization for metrics before backward. A non-finite
    metric prevents the optimizer step just as a non-finite gradient does.

    Returns ``(stepped, total_norm, bad_gradients, bad_value_count,
    recovered_norm_overflow, host_metric_values)``. The detailed tensor scan
    and FP64 norm only run on the exceptional non-finite-total-norm path.
    """
    net = network.module if hasattr(network, "module") else network
    named_parameters = [
        (f"model.{name}", parameter)
        for name, parameter in net.named_parameters()
        if parameter.grad is not None
    ]
    if isinstance(criterion, nn.Module):
        named_parameters.extend(
            (f"criterion.{name}", parameter)
            for name, parameter in criterion.named_parameters()
            if parameter.grad is not None
        )

    parameters = [parameter for _, parameter in named_parameters]
    if not parameters:
        raise RuntimeError("No gradients were produced for the optimizer step")

    gradients = [parameter.grad.detach() for parameter in parameters]
    foreach_compatible = all(
        gradient.layout == torch.strided
        and gradient.device == gradients[0].device
        and gradient.dtype == gradients[0].dtype
        for gradient in gradients
    )
    if foreach_compatible:
        # A model can have hundreds of parameter tensors. The foreach path
        # computes their norms with a fused multi-tensor kernel instead of one
        # kernel launch per tensor on every optimizer step.
        per_tensor_norms = torch._foreach_norm(gradients, 2.0)
    else:
        per_tensor_norms = [
            torch.linalg.vector_norm(gradient, 2) for gradient in gradients
        ]
    total_norm = torch.linalg.vector_norm(torch.stack(per_tensor_norms), 2)
    # This is the one required host synchronization in the ordinary path: the
    # BF16 guard must know whether stepping is safe. Fold scalar metrics into
    # the same small transfer so training does not stall once before backward
    # and once again here.
    if metric_values is None:
        host_values = [float(total_norm.detach())]
        host_metric_values = None
    else:
        if metric_values.ndim != 1:
            raise ValueError("metric_values must be a one-dimensional tensor")
        synchronized_values = torch.cat((
            total_norm.detach().float().reshape(1),
            metric_values.detach().float(),
        ))
        host_values = synchronized_values.cpu().tolist()
        host_metric_values = host_values[1:]
    total_norm_value = host_values[0]
    metrics_are_finite = (
        host_metric_values is None
        or all(math.isfinite(value) for value in host_metric_values)
    )
    recovered_norm_overflow = False
    bad_gradients = {}
    bad_value_count = 0

    if not math.isfinite(total_norm_value):
        for name, parameter in named_parameters:
            finite = torch.isfinite(parameter.grad)
            count = int((~finite).sum().item())
            if count:
                bad_gradients[name] = count
                bad_value_count += count

        if not bad_gradients:
            # Every element is finite, so the FP32 sum of squares—not the
            # backward pass—overflowed. Recompute only this exceptional path
            # in FP64 before clipping; the largest possible FP32 gradient set
            # remains far inside FP64's dynamic range.
            total_squared = torch.zeros(
                (), dtype=torch.float64, device=gradients[0].device
            )
            for gradient in gradients:
                total_squared.add_(gradient.double().square().sum())
            total_norm = total_squared.sqrt()
            total_norm_value = float(total_norm.detach())
            recovered_norm_overflow = True

    if bad_gradients or not metrics_are_finite:
        # GradScaler records found-inf state during unscale_. Updating without
        # stepping lowers its scale exactly as an automatically skipped FP16
        # step would. BF16 uses no scaler and simply discards the gradients.
        if scaler is not None and scaler.is_enabled():
            scaler.update()
        optimizer.zero_grad(set_to_none=True)
        return (
            False,
            total_norm_value,
            bad_gradients,
            bad_value_count,
            False,
            host_metric_values,
        )

    if clip_grads is not None:
        clip_coefficient = clip_grads / (total_norm + 1e-6)
        clip_coefficient = torch.clamp(clip_coefficient, max=1.0)
        with torch.no_grad():
            clip_coefficient = clip_coefficient.to(
                device=gradients[0].device, dtype=gradients[0].dtype
            )
            if foreach_compatible:
                torch._foreach_mul_(gradients, clip_coefficient)
            else:
                for gradient in gradients:
                    gradient.mul_(clip_coefficient.to(
                        device=gradient.device, dtype=gradient.dtype
                    ))

    if scaler is not None and scaler.is_enabled():
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    return (
        True,
        total_norm_value,
        {},
        0,
        recovered_norm_overflow,
        host_metric_values,
    )


def build_adamw(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    extra_modules: Optional[Iterable[nn.Module]] = None,
) -> Tuple[torch.optim.AdamW, dict]:
    """Build AdamW groups without dropping specially named parameters.

    Matrix/tensor weights receive weight decay. Biases, normalization vectors,
    transition logits, layer scales, and parameters explicitly marked with
    ``_no_weight_decay`` do not.
    """
    decay, no_decay = [], []
    seen = set()

    modules = [("model", model)]
    if extra_modules:
        modules.extend((f"extra_{i}", module) for i, module in enumerate(extra_modules))

    for prefix, module in modules:
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            full_name = f"{prefix}.{name}"
            exclude_decay = (
                parameter.ndim < 2
                or full_name.endswith(".bias")
                or bool(getattr(parameter, "_no_weight_decay", False))
            )
            (no_decay if exclude_decay else decay).append(parameter)

    expected = {
        id(parameter): parameter
        for _, module in modules
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    missing = set(expected) - seen
    if missing:
        missing_count = sum(expected[param_id].numel() for param_id in missing)
        raise RuntimeError(f"Optimizer construction omitted {missing_count} trainable parameters")

    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )
    summary = {
        "decay_tensors": len(decay),
        "no_decay_tensors": len(no_decay),
        "decay_parameters": sum(parameter.numel() for parameter in decay),
        "no_decay_parameters": sum(parameter.numel() for parameter in no_decay),
        "total_parameters": sum(parameter.numel() for parameter in expected.values()),
    }
    return optimizer, summary
