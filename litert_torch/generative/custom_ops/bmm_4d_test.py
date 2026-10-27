# Copyright 2026 The LiteRT Torch Authors.
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
"""Tests that bmm_4d converts to a single BATCH_MATMUL with adj_y."""

import litert_torch
from litert_torch.generative.custom_ops import bmm_4d as bmm_lib
import numpy as np
import torch

from absl.testing import absltest as googletest
from absl.testing import parameterized
from ai_edge_litert import interpreter as tfl_interpreter  # pylint: disable=g-direct-tensorflow-import
from ai_edge_litert import schema_py_generated as schema_fb


_CASES = (
    # Query @ key^T, key as [B, N, KV_LEN, H]: the KV length is a row of rhs.
    dict(
        testcase_name="qk",
        lhs_shape=(1, 2, 3, 8),
        rhs_shape=(1, 2, 6, 8),
        lhs_kv_dim=None,
        rhs_kv_dim=2,
    ),
    # Probs @ value^T, value as [B, N, H, KV_LEN]: the KV length is contracted.
    dict(
        testcase_name="pv",
        lhs_shape=(1, 2, 3, 6),
        rhs_shape=(1, 2, 8, 6),
        lhs_kv_dim=3,
        rhs_kv_dim=3,
    ),
)


class Bmm4d(torch.nn.Module):

  def forward(self, lhs, rhs):
    return bmm_lib.bmm_4d(lhs, rhs)


def _inputs(lhs_shape, rhs_shape, lhs_kv_dim, rhs_kv_dim, kv_len=None):
  lhs_shape, rhs_shape = list(lhs_shape), list(rhs_shape)
  if kv_len is not None:
    if lhs_kv_dim is not None:
      lhs_shape[lhs_kv_dim] = kv_len
    rhs_shape[rhs_kv_dim] = kv_len
  return torch.randn(lhs_shape), torch.randn(rhs_shape)


def _op_names(edge_model) -> list[str]:
  interpreter = tfl_interpreter.Interpreter(
      model_content=edge_model.model_content()
  )
  return [op["op_name"] for op in interpreter._get_ops_details()]  # pylint: disable=protected-access


class Bmm4dTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    torch.manual_seed(0)

  def _assert_single_batch_matmul_adj_y(self, edge_model):
    self.assertEqual(_op_names(edge_model), ["BATCH_MATMUL"])
    model = schema_fb.ModelT.InitFromPackedBuf(edge_model.model_content(), 0)
    subgraphs = model.subgraphs
    if not subgraphs:
      self.fail("Expected a subgraph.")
    operators = subgraphs[0].operators
    if not operators:
      self.fail("Expected an operator.")
    options = operators[0].builtinOptions
    if not isinstance(options, schema_fb.BatchMatMulOptionsT):
      self.fail(f"Expected BatchMatMulOptions, got {type(options)}.")
    self.assertFalse(options.adjX)
    self.assertTrue(options.adjY)

  def _assert_matches_torch(self, model, edge_model, lhs, rhs):
    expected = model(lhs, rhs).detach().numpy()
    actual = edge_model(lhs.numpy(), rhs.numpy())
    if not isinstance(actual, np.ndarray):
      self.fail(f"Expected a single output array, got {type(actual)}.")
    self.assertEqual(actual.shape, expected.shape)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

  @parameterized.named_parameters(*_CASES)
  def test_static_shape(self, lhs_shape, rhs_shape, lhs_kv_dim, rhs_kv_dim):
    model = Bmm4d().eval()
    lhs, rhs = _inputs(lhs_shape, rhs_shape, lhs_kv_dim, rhs_kv_dim)
    edge_model = litert_torch.convert(model, (lhs, rhs))

    self._assert_single_batch_matmul_adj_y(edge_model)
    self._assert_matches_torch(model, edge_model, lhs, rhs)

  @parameterized.named_parameters(*_CASES)
  def test_dynamic_kv_len(self, lhs_shape, rhs_shape, lhs_kv_dim, rhs_kv_dim):
    model = Bmm4d().eval()
    kv_len = torch.export.Dim("kv_len", min=2, max=64)
    dynamic_shapes = (
        None if lhs_kv_dim is None else {lhs_kv_dim: kv_len},
        {rhs_kv_dim: kv_len},
    )
    lhs, rhs = _inputs(lhs_shape, rhs_shape, lhs_kv_dim, rhs_kv_dim)
    edge_model = litert_torch.convert(
        model, (lhs, rhs), dynamic_shapes=dynamic_shapes
    )

    self._assert_single_batch_matmul_adj_y(edge_model)
    for kv in (6, 11):
      lhs, rhs = _inputs(lhs_shape, rhs_shape, lhs_kv_dim, rhs_kv_dim, kv)
      self._assert_matches_torch(model, edge_model, lhs, rhs)


if __name__ == "__main__":
  googletest.main()
