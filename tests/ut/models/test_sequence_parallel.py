# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from vllm_ascend.models.common.ops import sequence_parallel


def test_can_use_sp_mm_reduce_scatter_for_unquantized_biasless_linear(
    monkeypatch,
):
    monkeypatch.setattr(
        sequence_parallel,
        "get_tensor_model_parallel_world_size",
        lambda: 4,
    )
    linear = SimpleNamespace(
        quant_method=UnquantizedLinearMethod(),
        bias=None,
        weight=torch.empty(2, 2, dtype=torch.bfloat16),
    )

    assert sequence_parallel.can_use_sp_mm_reduce_scatter(linear)

    linear.bias = torch.zeros(1)
    assert not sequence_parallel.can_use_sp_mm_reduce_scatter(linear)


def test_sp_mm_reduce_scatter_pads_tokens_and_passes_tp_group(monkeypatch):
    backend = MagicMock()
    backend.get_hccl_comm_name.return_value = "tp-hcom"
    tp_group = SimpleNamespace(
        world_size=4,
        device_group=SimpleNamespace(
            _get_backend=MagicMock(return_value=backend),
        ),
    )
    monkeypatch.setattr(sequence_parallel, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(
        sequence_parallel,
        "get_world_group",
        lambda: SimpleNamespace(local_rank=3),
    )

    expected = torch.randn(2, 2)
    mmrs = MagicMock(return_value=expected)
    monkeypatch.setattr(
        sequence_parallel.DeviceOperator,
        "npu_mm_reduce_scatter_base",
        mmrs,
    )

    x = torch.randn(5, 3)
    weight = torch.randn(2, 3)
    output = sequence_parallel.sp_mm_reduce_scatter(x, weight)

    assert output is expected
    padded_x, transposed_weight, hcom, world_size = mmrs.call_args.args
    assert padded_x.shape == (8, 3)
    torch.testing.assert_close(padded_x[:5], x)
    torch.testing.assert_close(padded_x[5:], torch.zeros(3, 3))
    torch.testing.assert_close(transposed_weight, weight.transpose(0, 1))
    assert hcom == "tp-hcom"
    assert world_size == 4
    assert mmrs.call_args.kwargs == {
        "reduce_op": "sum",
        "comm_mode": "aiv",
    }
    backend.get_hccl_comm_name.assert_called_once_with(3)
