# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""REGRESSION: DistributedOptimizer must produce leaf shard params for FP32 model params.

``_build_model_and_main_param_groups`` slices a shard view out of every model param. The
float16 branch does so via ``model_param.detach().view(-1)[...]``, but the FP32 branch used
a plain ``model_param.view(-1)[...]``, which is a *non-leaf* tensor because the model param
requires grad. ``HybridDeviceOptimizer`` (``optimizer_cpu_offload=True``) passes those shards
straight to ``torch.optim.Optimizer``, which rejects non-leaf tensors with
"can't optimize a non-leaf Tensor".

Reached whenever an otherwise BF16/FP16 model keeps some parameter in FP32 -- e.g. anything
marked with ``mark_keep_in_fp32`` -- and CPU optimizer offloading is enabled.
"""

import torch
import torch.nn as nn

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.transformer.module import mark_keep_in_fp32
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class _Fp32MarkedToyNet(nn.Module):
    """BF16 linear plus one parameter pinned to FP32."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8, bias=False).to(torch.bfloat16)
        self.scale = nn.Parameter(torch.ones(8, dtype=torch.float32))
        mark_keep_in_fp32(self.scale)

    def forward(self, x):
        return self.proj(x) * self.scale


def _build_model():
    config = TransformerConfig(
        num_layers=1, hidden_size=8, num_attention_heads=1, bf16=True, params_dtype=torch.bfloat16
    )
    ddp_config = DistributedDataParallelConfig(
        use_distributed_optimizer=True, overlap_grad_reduce=False
    )
    return [DistributedDataParallel(config, ddp_config, _Fp32MarkedToyNet().cuda())]


def _optimizer_config(cpu_offload: bool) -> OptimizerConfig:
    return OptimizerConfig(
        optimizer="adam",
        lr=1e-4,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_distributed_optimizer=True,
        optimizer_cpu_offload=cpu_offload,
        optimizer_offload_fraction=1.0 if cpu_offload else 0.0,
        use_precision_aware_optimizer=cpu_offload,
    )


class TestDistribOptimizerFp32ParamShards:
    def setup_method(self, method):
        Utils.initialize_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_fp32_param_shards_are_leaves(self):
        """The shard views handed to the inner optimizer must be leaf tensors."""
        model = _build_model()
        optimizer = get_megatron_optimizer(_optimizer_config(cpu_offload=False), model)
        for group in optimizer.optimizer.param_groups:
            for param in group["params"]:
                assert param.is_leaf, "shard param handed to the inner optimizer is non-leaf"

    def test_cpu_offload_builds_with_fp32_param(self):
        """HybridDeviceOptimizer construction must not raise on FP32 model params."""
        model = _build_model()
        assert any(p.dtype == torch.float32 for p in model[0].parameters())
        get_megatron_optimizer(_optimizer_config(cpu_offload=True), model)
