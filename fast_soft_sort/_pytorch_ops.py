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

"""Numpy operators for soft sorting and ranking.
Fast Differentiable Sorting and Ranking
Mathieu Blondel, Olivier Teboul, Quentin Berthet, Josip Djolonga
https://arxiv.org/abs/2002.08871
This implementation follows the notation of the paper whenever possible.
"""

from .third_party import isotonic_pytorch
import torch

softmax = torch.nn.Softmax()


def isotonic_l2(input_s, input_w=None):
  """Solves an isotonic regression problem using PAV.
  Formally, it solves argmin_{v_1 >= ... >= v_n} 0.5 ||v - (s-w)||^2.
  Args:
    input_s: input to isotonic regression, a 1d-array.
    input_w: input to isotonic regression, a 1d-array.
  Returns:
    solution to the optimization problem.
  """
  if input_w is None:
    input_w = torch.arange(start=input_s.shape[0], end=0, step=-1,
                           dtype=input_s.dtype, device=input_s.device)
  solution = torch.zeros_like(input_s)
  isotonic_pytorch.isotonic_l2(input_s - input_w, solution)
  return solution


def isotonic_kl(input_s, input_w=None):
  """Solves isotonic optimization with KL divergence using PAV.

  Formally, it solves argmin_{v_1 >= ... >= v_n} <e^{s-v}, 1> + <e^w, v>.

  Args:
    input_s: input to isotonic optimization, a 1d-array.
    input_w: input to isotonic optimization, a 1d-array.
  Returns:
    solution to the optimization problem (same dtype as input_s).
  """
  if input_w is None:
    input_w = torch.arange(start=input_s.shape[0], end=0, step=-1,
                           dtype=input_s.dtype, device=input_s.device)
  solution = torch.zeros_like(input_s)
  isotonic_pytorch.isotonic_kl(input_s, input_w, solution)
  return solution


def _partition(solution, eps=1e-9):
  """Returns partition corresponding to solution."""
  # pylint: disable=g-explicit-length-test
  if len(solution) == 0:
    return []

  sizes = [1]

  for i in range(1, len(solution)):
    if abs(solution[i] - solution[i - 1]) > eps:
      sizes.append(0)
    sizes[-1] += 1

  return sizes


def _check_regularization(regularization):
  if regularization not in ("l2", "kl"):
    raise ValueError("'regularization' should be either 'l2' or 'kl' "
                     "but got %s." % str(regularization))


class _Differentiable(object):
  """Base class for differentiable operators."""

  def jacobian(self):
    """Computes Jacobian."""
    identity = torch.eye(self.size, device=self.device, dtype=self.dtype)
    jacobian = \
      torch.stack([self.jvp(identity[i]) for i in range(len(identity))]).T
    return jacobian

  @property
  def size(self):
    raise NotImplementedError

  @property
  def device(self):
    raise NotImplementedError

  @property
  def dtype(self):
    raise NotImplementedError

  def compute(self):
    """Computes the desired quantity."""
    raise NotImplementedError

  def jvp(self, vector):
    """Computes Jacobian vector product."""
    raise NotImplementedError

  def vjp(self, vector):
    """Computes vector Jacobian product."""
    raise NotImplementedError


class Isotonic(_Differentiable):
  """Isotonic optimization."""

  def __init__(self, input_s, input_w, regularization="l2"):
    self.input_s = input_s
    self.input_w = input_w
    _check_regularization(regularization)
    self.regularization = regularization
    self.solution_ = None

  @property
  def size(self):
    return self.input_s.shape[0]

  @property
  def device(self):
    return self.input_s.device

  @property
  def dtype(self):
    return self.input_s.dtype

  def compute(self):

    if self.regularization == "l2":
      self.solution_ = isotonic_l2(self.input_s, self.input_w)
    else:
      self.solution_ = isotonic_kl(self.input_s, self.input_w)
    return self.solution_

  def _check_computed(self):
    if self.solution_ is None:
      raise RuntimeError("Need to run compute() first.")

  def jvp(self, vector):
    self._check_computed()
    start = 0
    return_value = torch.zeros_like(self.solution_)
    for size in _partition(self.solution_):
      end = start + size
      if self.regularization == "l2":
        val = torch.mean(vector[start:end])
      else:
        val = torch.dot(softmax(self.input_s[start:end]),
                        vector[start:end])
      return_value[start:end] = val
      start = end
    return return_value

  def vjp(self, vector):
    start = 0
    return_value = torch.zeros_like(self.solution_)
    for size in _partition(self.solution_):
      end = start + size
      if self.regularization == "l2":
        val = 1. / size
      else:
        val = softmax(self.input_s[start:end])
      return_value[start:end] = val * torch.sum(vector[start:end])
      start = end
    return return_value


def _inv_permutation(permutation):
  """Returns inverse permutation of 'permutation'."""
  inv_permutation = torch.zeros_like(permutation)
  inv_permutation[permutation] = torch.arange(permutation.shape[0],
                                              device=permutation.device)
  return inv_permutation


class Projection(_Differentiable):
  """Computes projection onto the permutahedron P(w)."""

  def __init__(self, input_theta, input_w=None, regularization="l2"):
    self.input_theta = input_theta
    if input_w is None:
      input_w = torch.arange(start=input_theta.shape[0], end=0, step=-1,
                             dtype=input_theta.dtype, device=input_theta.device)
    self.input_w = input_w
    _check_regularization(regularization)
    self.regularization = regularization
    self.isotonic = None

  def _check_computed(self):
    if self.isotonic_ is None:
      raise ValueError("Need to run compute() first.")

  @property
  def size(self):
    return self.input_theta.shape[0]

  @property
  def device(self):
    return self.input_theta.device

  @property
  def dtype(self):
    return self.input_theta.dtype

  def compute(self):
    self.permutation = torch.argsort(self.input_theta, descending=True)
    input_s = self.input_theta[self.permutation]

    self.isotonic_ = Isotonic(input_s, self.input_w, self.regularization)
    dual_sol = self.isotonic_.compute()
    primal_sol = input_s - dual_sol

    self.inv_permutation = _inv_permutation(self.permutation)
    return primal_sol[self.inv_permutation]

  def jvp(self, vector):
    self._check_computed()
    ret = vector - \
          self.isotonic_.jvp(vector[self.permutation])[self.inv_permutation]
    return ret

  def vjp(self, vector):
    self._check_computed()
    ret = vector - \
          self.isotonic_.vjp(vector[self.permutation])[self.inv_permutation]
    return ret


def _check_direction(direction):
  if direction not in ("ASCENDING", "DESCENDING"):
    raise ValueError("direction should be either 'ASCENDING' or 'DESCENDING'")


class SoftRank(_Differentiable):
  """Soft ranking."""

  def __init__(self, values, direction="ASCENDING",
               regularization_strength=1.0, regularization="l2"):
    if type(values) is list:
      self.values = torch.tensor(values)
    else:
      self.values = values
    self.input_w = torch.arange(start=self.values.shape[0], end=0, step=-1,
                                dtype=self.values.dtype,
                                device=self.values.device)
    _check_direction(direction)
    sign = 1 if direction == "ASCENDING" else -1
    self.scale = sign / regularization_strength
    _check_regularization(regularization)
    self.regularization = regularization
    self.projection_ = None

  @property
  def size(self):
    return self.values.shape[0]

  @property
  def device(self):
    return self.values.device

  @property
  def dtype(self):
    return self.values.dtype

  def _check_computed(self):
    if self.projection_ is None:
      raise ValueError("Need to run compute() first.")

  def compute(self):
    if self.regularization == "kl":
      self.projection_ = Projection(
          self.values * self.scale,
          torch.log(self.input_w),
          regularization=self.regularization)
      self.factor = torch.exp(self.projection_.compute())
      return self.factor
    else:
      self.projection_ = Projection(
          self.values * self.scale, self.input_w,
          regularization=self.regularization)
      self.factor = 1.0
      return self.projection_.compute()

  def jvp(self, vector):
    self._check_computed()
    return self.factor * self.projection_.jvp(vector) * self.scale

  def vjp(self, vector):
    self._check_computed()
    return self.projection_.vjp(self.factor * vector) * self.scale


class SoftSort(_Differentiable):
  """Soft sorting."""

  def __init__(self, values, direction="ASCENDING",
               regularization_strength=1.0, regularization="l2"):
    if type(values) is list:
      self.values = torch.tensor(values)
    else:
      self.values = values
    _check_direction(direction)
    self.sign = 1 if direction == "DESCENDING" else -1
    self.regularization_strength = regularization_strength
    _check_regularization(regularization)
    self.regularization = regularization
    self.isotonic_ = None

  @property
  def size(self):
    return self.values.shape[0]

  @property
  def device(self):
    return self.values.device

  @property
  def dtype(self):
    return self.values.dtype

  def _check_computed(self):
    if self.isotonic_ is None:
      raise ValueError("Need to run compute() first.")

  def compute(self):
    input_w = torch.arange(start=self.values.shape[0], end=0, step=-1,
                           dtype=self.values.dtype, device=self.values.device)\
                           / self.regularization_strength
    values = self.sign * self.values
    self.permutation_ = torch.argsort(values, descending=True)
    s = values[self.permutation_]

    self.isotonic_ = Isotonic(input_w, s, regularization=self.regularization)
    res = self.isotonic_.compute()

    # We set s as the first argument as we want the derivatives w.r.t. s.
    self.isotonic_.s = s
    return self.sign * (input_w - res)

  def jvp(self, vector):
    self._check_computed()
    return self.isotonic_.jvp(vector[self.permutation_])

  def vjp(self, vector):
    self._check_computed()
    inv_permutation = _inv_permutation(self.permutation_)
    return self.isotonic_.vjp(vector)[inv_permutation]


class Sort(_Differentiable):
  """Hard sorting."""

  def __init__(self, values, direction="ASCENDING"):
    _check_direction(direction)
    if type(values) is list:
      self.values = torch.tensor(values)
    else:
      self.values = values
    self.sign = 1 if direction == "DESCENDING" else -1
    self.permutation_ = None

  @property
  def size(self):
    return self.values.shape[0]

  @property
  def device(self):
    return self.values.device

  @property
  def device(self):
    return self.dtype.device

  def _check_computed(self):
    if self.permutation_ is None:
      raise ValueError("Need to run compute() first.")

  def compute(self):
    self.permutation_ = torch.argsort(self.sign * self.values, descending=True)
    return self.values[self.permutation_]

  def jvp(self, vector):
    self._check_computed()
    return vector[self.permutation_]

  def vjp(self, vector):
    self._check_computed()
    inv_permutation = _inv_permutation(self.permutation_)
    return vector[inv_permutation]


# Small utility functions for the case when we just want the forward
# computation.


def soft_rank(values, direction="ASCENDING", regularization_strength=1.0,
              regularization="l2"):
  r"""Soft rank the given values.
  The regularization strength determines how close are the returned values
  to the actual ranks.
  Args:
    values: A 1d-array holding the numbers to be ranked.
    direction: Either 'ASCENDING' or 'DESCENDING'.
    regularization_strength: The regularization strength to be used. The smaller
    this number, the closer the values to the true ranks.
    regularization: Which regularization method to use. It
      must be set to one of ("l2", "kl", "log_kl").
  Returns:
    A 1d-array, soft-ranked.
  """
  return SoftRank(values, regularization_strength=regularization_strength,
                  direction=direction, regularization=regularization).compute()


def soft_sort(values, direction="ASCENDING", regularization_strength=1.0,
              regularization="l2"):
  r"""Soft sort the given values.
  Args:
    values: A 1d-array holding the numbers to be sorted.
    direction: Either 'ASCENDING' or 'DESCENDING'.
    regularization_strength: The regularization strength to be used. The smaller
    this number, the closer the values to the true sorted values.
    regularization: Which regularization method to use. It
      must be set to one of ("l2", "log_kl").
  Returns:
    A 1d-array, soft-sorted.
  """
  return SoftSort(values, regularization_strength=regularization_strength,
                  direction=direction, regularization=regularization).compute()


def sort(values, direction="ASCENDING"):
  r"""Sort the given values.
  Args:
    values: A 1d-array holding the numbers to be sorted.
    direction: Either 'ASCENDING' or 'DESCENDING'.
  Returns:
    A 1d-array, sorted.
  """
  return Sort(values, direction=direction).compute()


def rank(values, direction="ASCENDING"):
  r"""Rank the given values.
  Args:
    values: A 1d-array holding the numbers to be ranked.
    direction: Either 'ASCENDING' or 'DESCENDING'.
  Returns:
    A 1d-array, ranked.
  """
  permutation = None
  if direction == "DESCENDING":
    permutation = torch.argsort(values, descending=True)
  else:
    permutation = torch.argsort(values)
  return _inv_permutation(permutation) + 1  # We use 1-based indexing.
