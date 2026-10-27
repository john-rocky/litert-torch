# Copyright 2024 The LiteRT Torch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Common utility functions for data loading etc.
from dataclasses import dataclass
from litert_torch.backend import lowerings
from litert_torch.backend.lowerings import utils as lowering_utils
from litert_torch.backend.torch_tfl import _lowerings as tfl_lowerings
from litert_converter.mlir import ir
import torch


# Use torch.library.custom_op to define a new custom operator.
@torch.library.custom_op("litert_torch::bmm_4d", mutates_args=())
def bmm_4d(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
  if not (lhs.ndim == 4 and rhs.ndim == 4):
    raise ValueError("bmm_4d requires LHS and RHS have rank 4.")
  d0_can_bcast = (
      lhs.shape[0] == rhs.shape[0] or lhs.shape[0] == 1 or rhs.shape[0] == 1
  )
  d1_can_bcast = (
      lhs.shape[1] == rhs.shape[1] or lhs.shape[1] == 1 or rhs.shape[1] == 1
  )
  if not (d0_can_bcast and d1_can_bcast):
    raise ValueError("bmm_4d requires that dimensions 0 and 1 can broadcast.")

  if not lhs.shape[-1] == rhs.shape[-1]:
    raise ValueError("bmm_4d requires LHS and RHS have same last dimension.")

  return torch.einsum("abcd,abed->abce", lhs, rhs)


# Use register_fake to add a ``FakeTensor`` kernel for the operator
@bmm_4d.register_fake
def _(lhs, rhs):
  return torch.einsum("abcd,abed->abce", lhs, rhs)


# Lowers to tfl.batch_matmul with adj_y, i.e. lhs @ rhs^T on the last two dims.
# The converter folds an equivalent stablehlo.dot_general into the same
# BATCH_MATMUL when the shapes are static, but with a dynamic KV cache length
# (export with dynamic shapes) it expands it into a TRANSPOSE plus shape
# arithmetic. The result type comes from the node meta, so dynamic dimensions
# pass through.
@lowerings.lower(torch.ops.litert_torch.bmm_4d)
def _bmm_4d_lower(
    lctx,
    lhs: ir.Value,
    rhs: ir.Value,
):
  return tfl_lowerings._ir_operation(  # pylint: disable=protected-access
      "tfl.batch_matmul",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[lhs, rhs],
      attributes={
          "adj_x": ir.BoolAttr.get(False),
          "adj_y": ir.BoolAttr.get(True),
          "asymmetric_quantize_inputs": ir.BoolAttr.get(False),
      },
  )
