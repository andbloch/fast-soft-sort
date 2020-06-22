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

"""Tests for isotonic.py."""

import unittest
from absl.testing import parameterized
from fast_soft_sort.third_party import isotonic_numpy
from fast_soft_sort.third_party import isotonic_pytorch
import numpy as np
import torch
from sklearn.isotonic import isotonic_regression


class IsotonicTest(parameterized.TestCase):

  def test_l2_agrees_with_sklearn(self):
    rng = np.random.RandomState(0)

    y_numpy = rng.randn(10) * rng.randint(1, 5)
    y_pytorch = torch.from_numpy(y_numpy)

    sol_numpy = np.zeros_like(y_numpy)
    isotonic_numpy.isotonic_l2(y_numpy, sol_numpy)

    sol_pytorch = torch.zeros_like(y_pytorch)
    isotonic_pytorch.isotonic_l2(y_pytorch, sol_pytorch)

    sol_skl = isotonic_regression(y_numpy, increasing=False)

    np.testing.assert_array_almost_equal(sol_skl, sol_numpy)
    np.testing.assert_array_almost_equal(sol_skl, sol_pytorch)
    np.testing.assert_array_almost_equal(sol_pytorch, sol_numpy)

  def test_kl_pytorch_agrees_with_kl_numpy(self):
    rng = np.random.RandomState(0)

    y_numpy = rng.randn(10) * rng.randint(1, 5)
    y_pytorch = torch.from_numpy(y_numpy)

    w_numpy = np.array(sorted(rng.randn(y_numpy.shape[0])))
    w_pytorch = torch.from_numpy(w_numpy)

    sol_numpy = np.zeros_like(y_numpy)
    isotonic_numpy.isotonic_kl(y_numpy, w_numpy, sol_numpy)

    sol_pytorch = torch.zeros_like(y_pytorch)
    isotonic_pytorch.isotonic_kl(y_pytorch, w_pytorch, sol_pytorch)

    np.testing.assert_array_almost_equal(sol_numpy, sol_pytorch)



if __name__ == "__main__":
  unittest.main()
