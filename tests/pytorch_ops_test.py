# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for _pytorch_ops.py."""

import itertools
import unittest
from absl.testing import parameterized
from fast_soft_sort import _pytorch_ops as ops
import numpy as np
import torch

# TODO: eps has to be chosen very small here! TODO: Fix numerics!
# TODO: eps has to be chosen very small here! TODO: Fix numerics!
# TODO: eps has to be chosen very small here! TODO: Fix numerics!
def _num_jacobian(theta, f, eps=1e-3):
  n_classes = len(theta)
  ret = torch.zeros((n_classes, n_classes),
                    dtype=theta.dtype, device=theta.device)

  for i in range(n_classes):
    theta_ = theta.clone()
    theta_[i] += eps
    val = f(theta_)
    theta_[i] -= 2 * eps
    val2 = f(theta_)
    ret[i] = (val - val2) / (2 * eps)

  return ret.T


GAMMAS = (0.1, 1.0, 10.0)
DIRECTIONS = ("ASCENDING", "DESCENDING")
REGULARIZERS = ("l2", "kl")
CLASSES = (ops.Isotonic, ops.Projection)
DTYPES = (torch.float64,)


class IsotonicProjectionTest(parameterized.TestCase):

  @parameterized.parameters(itertools.product(CLASSES, REGULARIZERS, DTYPES))
  def test_jvp_and_vjp_against_numerical_jacobian(self, cls, regularization,
                                                  dtype):
    torch.manual_seed(0)
    theta = torch.randn(5, dtype=dtype)
    w = torch.arange(start=5, end=0, step=-1, dtype=theta.dtype)
    v = torch.randn(5, dtype=dtype)

    f = lambda x: cls(x, w, regularization=regularization).compute()
    J = _num_jacobian(theta, f)

    obj = cls(theta, w, regularization=regularization)
    obj.compute()

    out = obj.jvp(v)
    target = torch.matmul(J, v)
    np.testing.assert_array_almost_equal(target, out)

    out = obj.vjp(v)
    target = torch.matmul(v, J)
    np.testing.assert_array_almost_equal(target, out)


class SoftRankTest(parameterized.TestCase):

  @parameterized.parameters(itertools.product(DIRECTIONS, REGULARIZERS, DTYPES))
  def test_soft_rank_converges_to_hard(self, direction, regularization, dtype):
    torch.manual_seed(0)
    theta = torch.randn(5, dtype=dtype)
    soft_rank = ops.SoftRank(theta, regularization_strength=1e-3,
                             direction=direction, regularization=regularization)
    out = ops.rank(theta, direction=direction)
    out2 = soft_rank.compute()
    np.testing.assert_array_almost_equal(out, out2)

  @parameterized.parameters(itertools.product(GAMMAS, DIRECTIONS, REGULARIZERS,
                                              DTYPES))
  def test_soft_rank_jvp_and_vjp_against_numerical_jacobian(self,
                                                            regularization_strength,
                                                            direction,
                                                            regularization,
                                                            dtype):
    torch.manual_seed(0)
    theta = torch.randn(5, dtype=dtype)
    v = torch.randn(5, dtype=dtype)

    f = lambda x: ops.SoftRank(x,
        regularization_strength=regularization_strength, direction=direction,
        regularization=regularization).compute()
    J = _num_jacobian(theta, f)

    soft_rank = ops.SoftRank(theta,
                             regularization_strength=regularization_strength,
                             direction=direction, regularization=regularization)
    soft_rank.compute()

    out = soft_rank.jvp(v)
    target = torch.matmul(J, v)
    np.testing.assert_array_almost_equal(target, out, 1e-6)

    out = soft_rank.vjp(v)
    target = torch.matmul(v, J)
    np.testing.assert_array_almost_equal(target, out, 1e-6)

    out = soft_rank.jacobian()
    np.testing.assert_array_almost_equal(J, out, 1e-6)

  @parameterized.parameters(itertools.product(GAMMAS, DIRECTIONS, REGULARIZERS,
                                              DTYPES))
  def test_soft_rank_works_with_lists(self, regularization_strength, direction,
                                      regularization, dtype):
    torch.manual_seed(0)
    theta = torch.randn(5, dtype=dtype).to(torch.float32).to(dtype)
    ranks1 = ops.SoftRank(theta,
                          regularization_strength=regularization_strength,
                          direction=direction,
                          regularization=regularization).compute()
    ranks2 = ops.SoftRank(theta.tolist(),
                          regularization_strength=regularization_strength,
                          direction=direction,
                          regularization=regularization).compute()
    np.testing.assert_array_almost_equal(ranks1, ranks2)


class SoftSortTest(parameterized.TestCase):

  @parameterized.parameters(itertools.product(DIRECTIONS, REGULARIZERS, DTYPES))
  def test_soft_sort_converges_to_hard(self, direction, regularization, dtype):
    torch.manual_seed(0)
    theta = torch.randn(5, dtype=dtype)
    soft_sort = ops.SoftSort(theta, regularization_strength=1e-3,
                             direction=direction, regularization=regularization)
    sort = ops.Sort(theta, direction=direction)
    out = sort.compute()
    out2 = soft_sort.compute()
    np.testing.assert_array_almost_equal(out, out2)

  @parameterized.parameters(itertools.product(GAMMAS, DIRECTIONS, REGULARIZERS,
                                              DTYPES))
  def test_soft_sort_jvp(self, regularization_strength, direction,
                         regularization, dtype):
    torch.manual_seed(0)
    theta = torch.randn(5, dtype=dtype)
    v = torch.randn(5, dtype=dtype)

    f = lambda x: ops.SoftSort(
        x, regularization_strength=regularization_strength,
        direction=direction, regularization=regularization).compute()
    J = _num_jacobian(theta, f)

    soft_sort = ops.SoftSort(theta,
        regularization_strength=regularization_strength,
        direction=direction, regularization=regularization)
    soft_sort.compute()

    out = soft_sort.jvp(v)
    np.testing.assert_array_almost_equal(torch.matmul(J, v), out, 1e-6)

    out = soft_sort.vjp(v)
    np.testing.assert_array_almost_equal(torch.matmul(v, J), out, 1e-6)

    out = soft_sort.jacobian()
    np.testing.assert_array_almost_equal(J, out, 1e-6)

  # TODO: fix numerics in this thest!
  @parameterized.parameters(itertools.product(GAMMAS, DIRECTIONS, REGULARIZERS,
                                              DTYPES))
  def test_soft_sort_works_with_lists(self, regularization_strength, direction,
                                      regularization, dtype):
    torch.manual_seed(0)
    theta = torch.randn(5, dtype=dtype).to(torch.float32).to(dtype)
    sort1 = ops.SoftSort(theta,
                         regularization_strength=regularization_strength,
                         direction=direction,
                         regularization=regularization).compute()
    sort2 = ops.SoftSort(theta.tolist(),
                         regularization_strength=regularization_strength,
                         direction=direction,
                         regularization=regularization).compute()
    np.testing.assert_array_almost_equal(sort1, sort2)


class SortTest(parameterized.TestCase):

  @parameterized.parameters(itertools.product(DIRECTIONS, DTYPES))
  def test_sort_jvp(self, direction, dtype):
    torch.manual_seed(0)
    theta = torch.randn(5, dtype=dtype)
    v = torch.randn(5, dtype=dtype)

    f = lambda x: ops.Sort(x, direction=direction).compute()
    J = _num_jacobian(theta, f)

    sort = ops.Sort(theta, direction=direction)
    sort.compute()

    out = sort.jvp(v)
    np.testing.assert_array_almost_equal(torch.matmul(J, v), out)

    out = sort.vjp(v)
    np.testing.assert_array_almost_equal(torch.matmul(v, J), out)

  @parameterized.parameters(itertools.product(DIRECTIONS, DTYPES))
  def test_sort_works_with_lists(self, direction, dtype):
    torch.manual_seed(0)
    theta = torch.randn(5, dtype=dtype).to(torch.float32).to(dtype)
    sort_pt = ops.Sort(theta, direction=direction).compute()
    sort_list = ops.Sort(theta.tolist(), direction=direction).compute()
    np.testing.assert_array_almost_equal(sort_pt, sort_list)


if __name__ == "__main__":
  unittest.main()
