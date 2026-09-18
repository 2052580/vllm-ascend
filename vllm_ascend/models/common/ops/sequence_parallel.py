# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    get_world_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from vllm_ascend.device.device_op import DeviceOperator


def _custom_collective(name: str, x: torch.Tensor) -> torch.Tensor | None:
    device_communicator = get_tp_group().device_communicator
    if device_communicator is None:
        return None
    collective = getattr(device_communicator, name, None)
    return None if collective is None else collective(x)


def sp_all_gather(x: torch.Tensor) -> torch.Tensor:
    output = _custom_collective("custom_all_gather", x)
    if output is not None:
        return output
    return tensor_model_parallel_all_gather(x, 0)


def sp_reduce_scatter(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2
    tp_size = get_tensor_model_parallel_world_size()
    sp_pad = (-x.shape[0]) % tp_size
    pad_shape = [sp_pad, x.shape[1]]
    x = torch.cat([x, x.new_zeros(pad_shape)], dim=0)
    output = _custom_collective("custom_reduce_scatter", x)
    if output is not None:
        return output
    return tensor_model_parallel_reduce_scatter(x, 0)


def can_use_sp_mm_reduce_scatter(linear: torch.nn.Module) -> bool:
    weight = getattr(linear, "weight", None)
    return (
        get_tensor_model_parallel_world_size() > 1
        and isinstance(getattr(linear, "quant_method", None), UnquantizedLinearMethod)
        and getattr(linear, "bias", None) is None
        and weight is not None
        and weight.dtype in (torch.float16, torch.bfloat16)
    )


def sp_mm_reduce_scatter(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Fuse a row-parallel BF16/FP16 matmul with sequence reduce-scatter."""
    assert x.ndim == 2
    assert weight.ndim == 2
    tp_group = get_tp_group()
    tp_size = tp_group.world_size
    if not x.is_contiguous():
        x = x.contiguous()
    sp_pad = (-x.shape[0]) % tp_size
    if sp_pad:
        x = torch.cat([x, x.new_zeros((sp_pad, x.shape[1]))], dim=0)

    backend = tp_group.device_group._get_backend(x.device)
    hcom = backend.get_hccl_comm_name(get_world_group().local_rank)
    return DeviceOperator.npu_mm_reduce_scatter_base(
        x,
        weight.transpose(0, 1),
        hcom,
        tp_size,
        reduce_op="sum",
        comm_mode="aiv",
    )


def sp_shard(x: torch.Tensor) -> torch.Tensor:
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    sp_pad = (-x.shape[0]) % tp_size
    pad_shape = list(x.shape)
    pad_shape[0] = sp_pad
    x = torch.cat([x, x.new_zeros(pad_shape)], dim=0)
    chunk = x.shape[0] // tp_size
    return x[tp_rank * chunk : (tp_rank + 1) * chunk]


def sp_padding_mask(
    is_padding: torch.Tensor | None,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]
    if is_padding is None:
        is_padding = hidden_states.new_zeros(num_tokens, dtype=torch.bool)
    assert is_padding.shape[0] == num_tokens

    tp_size = get_tensor_model_parallel_world_size()
    sp_pad = (-num_tokens) % tp_size
    is_padding = torch.cat([is_padding, is_padding.new_ones((sp_pad,))], dim=0)
    chunk = is_padding.shape[0] // tp_size
    tp_rank = get_tensor_model_parallel_rank()
    return is_padding[tp_rank * chunk : (tp_rank + 1) * chunk]
