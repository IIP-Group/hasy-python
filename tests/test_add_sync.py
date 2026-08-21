import numpy as np

from random_coefs import random_coef_generator
from add_sync_export import versinc, prerotate
from add_sync_export import transform_hard, transform_mirrored, transform_pulsar
from add_sync_export import FourierOscillator
from add_sync_export import versinc_vectorized, prerotate_vectorized
from add_sync_export import transform_hard_vectorized, transform_mirrored_vectorized, transform_pulsar_vectorized
from add_sync_export import FourierOscillator_vectorized


# --------------------------------- prerotate -------------------------------- #
def test_prerotate_zero_rotation():
  a0 = 1.0
  a = np.array([1.0, 2.0, 3.0, 4.0])
  b = np.array([0.5, 1.5, 2.5, 3.5])

  a0_rot, a_rot, b_rot = prerotate((a0, a, b), tau=0.0)

  assert a0_rot == a0
  np.testing.assert_allclose(a_rot, a)
  np.testing.assert_allclose(b_rot, b)

def test_prerotate_full_periods():
  a0 = 1.0
  a = np.array([1.0, 2.0, 3.0, 4.0])
  b = np.array([0.5, 1.5, 2.5, 3.5])

  for tau in range(-10, 10):
    a0_rot, a_rot, b_rot = prerotate((a0, a, b), tau=tau)

    assert a0_rot == a0
    np.testing.assert_allclose(a_rot, a, atol=1e-10)
    np.testing.assert_allclose(b_rot, b, atol=1e-10)


def test_prerotate_90deg():
  a0 = 0
  a = np.array([1.0, 1.0, 1.0, 1.0])
  b = np.array([0.0, 0.0, 0.0, 0.0])

  # + 90 deg
  a0_rot, a_rot, b_rot = prerotate((a0, a, b), tau=0.25)
  np.testing.assert_allclose(a0_rot, a0)
  np.testing.assert_allclose(a_rot, np.array([0, -1, 0, 1]), atol=1e-10)
  np.testing.assert_allclose(b_rot, np.array([1, 0, -1, 0]), atol=1e-10)

  # + 180 deg
  a0_rot, a_rot, b_rot = prerotate((a0, a, b), tau=0.5)
  np.testing.assert_allclose(a0_rot, a0)
  np.testing.assert_allclose(a_rot, np.array([-1, 1, -1, 1]), atol=1e-10)
  np.testing.assert_allclose(b_rot, np.array([0, 0, 0, 0]), atol=1e-10)

  # + 270 deg
  a0_rot, a_rot, b_rot = prerotate((a0, a, b), tau=0.75)
  np.testing.assert_allclose(a0_rot, a0)
  np.testing.assert_allclose(a_rot, np.array([ 0, -1, 0, 1]), atol=1e-10)
  np.testing.assert_allclose(b_rot, np.array([-1, 0, 1,  0]), atol=1e-10)


# ------------------------------ transform hard ------------------------------ #

def test_transform_hard_zero_input():
  N_max = 4
  coefs = (0.0, np.zeros(N_max), np.zeros(N_max))
  a0, a, b = transform_hard(coefs, period_ratio=1.0)
  assert a0 == 0.0
  np.testing.assert_allclose(a, np.zeros(N_max))
  np.testing.assert_allclose(b, np.zeros(N_max))

def test_transform_hard_identity():
  a0 = 1.0
  a = np.array([1.0, 2.0, 3.0, 4.0])
  b = np.array([0.5, 1.5, 2.5, 3.5])

  (result_a0, result_a, result_b) = transform_hard((a0, a, b), period_ratio=1.0)

  np.testing.assert_allclose(result_a0, a0, atol=1e-10)
  np.testing.assert_allclose(result_a,  a,  atol=1e-10)
  np.testing.assert_allclose(result_b,  b,  atol=1e-10)


# ------------------------------ transform mirrored ------------------------------ #
def test_transform_mirrored_zero_input():
  N_max = 4
  coefs = (0.0, np.zeros(N_max), np.zeros(N_max))
  a0, a, b = transform_mirrored(coefs, period_ratio=1.0)
  assert a0 == 0.0
  np.testing.assert_allclose(a, np.zeros(N_max))
  np.testing.assert_allclose(b, np.zeros(N_max))

def test_transform_mirrored_identity_even():
  a0 = 1.0
  a = np.array([1.0, 2.0, 3.0, 4.0])
  b = np.array([0.0, 0.0, 0.0, 0.0])  # odd harmonics are zero

  (result_a0, result_a, result_b) = transform_mirrored((a0, a, b), period_ratio=0.5) # P=0.5 for same pitch

  np.testing.assert_allclose(result_a0, a0, atol=1e-10)
  np.testing.assert_allclose(result_a,  a,  atol=1e-10)
  np.testing.assert_allclose(result_b,  b,  atol=1e-10)


# ----------------------------- transform pulsar ----------------------------- #
def test_transform_pulsar_zero_input():
  N_max = 4
  coefs = (0.0, np.zeros(N_max), np.zeros(N_max))
  a0, a, b = transform_pulsar(coefs, period_ratio=1.0)
  assert a0 == 0.0
  np.testing.assert_allclose(a, np.zeros(N_max))
  np.testing.assert_allclose(b, np.zeros(N_max))

def test_transform_pulsar_identity():
  a0 = 0.0 # pulsar sync ignores DFC -> must be zero
  a = np.array([1.0, 2.0, 3.0, 4.0])
  b = np.array([0.5, 1.5, 2.5, 3.5])

  (result_a0, result_a, result_b) = transform_pulsar((a0, a, b), period_ratio=1.0)

  np.testing.assert_allclose(result_a0, a0, atol=1e-10)
  np.testing.assert_allclose(result_a,  a,  atol=1e-10)
  np.testing.assert_allclose(result_b,  b,  atol=1e-10)


# --------------------------- vectorized functions --------------------------- #


def test_versic_vectorized_reg():
  xs = np.array([0.0, 1e-15, 1e-9, -1e-9, 0.25, -0.25, 1.0, -1.0, 3.0, -3.0])
  ref = np.array([versinc(float(x)) for x in xs])
  vec = versinc_vectorized(xs)
  np.testing.assert_allclose(vec, ref, atol=0.0, rtol=0.0)

  # scalar behavior
  assert versinc_vectorized(0.0) == versinc(0.0)
  assert versinc_vectorized(0.25) == versinc(0.25)


def test_prerotate_vectorized_reg(regression_repeats):
  make_coefs = random_coef_generator(seed=0)
  for _ in range(regression_repeats):
    coefs_chk = make_coefs(512)
    for _tau in [0.0, 0.25, -0.5, 1.3]:
      ref = prerotate(coefs_chk, _tau)
      vec = prerotate_vectorized(coefs_chk, _tau)
      np.testing.assert_allclose(vec[0], ref[0])
      np.testing.assert_allclose(vec[1], ref[1], atol=1e-12, rtol=0)
      np.testing.assert_allclose(vec[2], ref[2], atol=1e-12, rtol=0)


def test_transform_hard_vectorized_reg(regression_repeats):
  make_coefs = random_coef_generator(seed=0)
  for _ in range(regression_repeats):
    coefs_chk = make_coefs(512)
    for _P in [0.5, 1.0, 11/8, 1.5, 2.0, 2.3456]:
      ref = transform_hard(coefs_chk, _P)
      vec = transform_hard_vectorized(coefs_chk, _P)
      np.testing.assert_allclose(vec[0], ref[0], atol=1e-10, rtol=0)
      np.testing.assert_allclose(vec[1], ref[1], atol=1e-10, rtol=0)
      np.testing.assert_allclose(vec[2], ref[2], atol=1e-10, rtol=0)


def test_transform_mirrored_vectorized_reg(regression_repeats):
  make_coefs = random_coef_generator(seed=0)
  for _ in range(regression_repeats):
    coefs_chk = make_coefs(512)
    for _P in [0.5, 1.0, 11/8, 1.5, 2.0, 2.3456]:
      ref = transform_mirrored(coefs_chk, _P)
      vec = transform_mirrored_vectorized(coefs_chk, _P)
      np.testing.assert_allclose(vec[0], ref[0], atol=1e-10, rtol=0)
      np.testing.assert_allclose(vec[1], ref[1], atol=1e-10, rtol=0)
      np.testing.assert_allclose(vec[2], ref[2], atol=0.0, rtol=0.0)


def test_transform_pulsar_vectorized_reg(regression_repeats):
  make_coefs = random_coef_generator(seed=0)
  for _ in range(regression_repeats):
    coefs_chk = make_coefs(512)
    for _P in [0.5, 1.0, 11/8, 1.5, 2.0, 2.3456]:
      ref = transform_pulsar(coefs_chk, _P)
      vec = transform_pulsar_vectorized(coefs_chk, _P)
      np.testing.assert_allclose(vec[0], ref[0], atol=1e-10, rtol=0)
      np.testing.assert_allclose(vec[1], ref[1], atol=1e-10, rtol=0)
      np.testing.assert_allclose(vec[2], ref[2], atol=1e-10, rtol=0)


def test_FourierOscillator_generate_vectorized(regression_repeats):
  fs_chk_Hz = 96000
  n_smpl_chk = 1000
  make_coefs = random_coef_generator(seed=0)
  for _ in range(regression_repeats):
    coefs_chk = make_coefs(512)
    for f_chk_Hz in [20, 94, 375, 1000, 12345]:
      add_synth_ref = FourierOscillator(fs_chk_Hz, 512)
      add_synth_ref.set_coefs(coefs_chk)
      add_synth_ref.set_f_fund(f_chk_Hz)
      add_synth_vec = FourierOscillator_vectorized(fs_chk_Hz, 512)
      add_synth_vec.set_coefs(coefs_chk)
      add_synth_vec.set_f_fund(f_chk_Hz)
      ref = add_synth_ref.generate(n_smpl_chk)
      vec = add_synth_vec.generate(n_smpl_chk)
      np.testing.assert_allclose(vec, ref, atol=1e-6, rtol=0)
