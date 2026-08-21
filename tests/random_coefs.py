from collections.abc import Callable

import numpy as np


def random_coef_generator(seed: int) -> Callable[[int, float], tuple[float, np.ndarray, np.ndarray]]:
  """Return ``make(n_max, a0=None)``; fixed ``seed``, successive calls draw new coefs."""
  rng = np.random.default_rng(seed)

  def make(n_max: int, a0: float = None) -> tuple[float, np.ndarray, np.ndarray]:
    if a0 is None: # only if user did not specify a0
      a0 = rng.uniform(-1,1)

    return (a0, rng.uniform(-1,1,n_max), rng.uniform(-1,1,n_max))

  return make
