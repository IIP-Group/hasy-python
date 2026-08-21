import numpy as np

# based on https://www.reddit.com/r/gifs/comments/ag6or3/send_this_to_your_loved_ones_for_valentines/
def coefs_heart(N_max: int):
  """Fourier coefficients for a synthetic heart-shaped oscillation.

  Returns:
    (a0, a, b): DC term and first 32 Fourier coefficients.
  """
  # synthesize time-domain signal
  x_lim = 1.85
  x = np.linspace(-x_lim, x_lim, int(2*x_lim*1000), endpoint=False)
  f = 15
  r_heart = np.power(np.abs(x),2/3) + (np.power(2*x_lim-np.power(x,2),1/2) * np.sin(f*np.pi*x))
  r_heart = r_heart * 0.35

  # compute Fourier coefficients
  c = np.fft.fft(r_heart)/len(x)
  assert 2*N_max <= len(c)
  # note: 2*f+2 coefficients would often be sufficient
  n = np.arange(1,N_max+1)
  a0 = np.real(c[0])
  a = np.real(c[n] + c[-n])
  b = np.real(1j* (c[n] - c[-n]))

  return (a0, a, b)


# based on https://www.syntorial.com/preset-recipe/darude-sandstorm-lead/
def coefs_sandstorm(N_max: int):
  """Fourier coefficients for Darude's Sandstorm lead waveform.

  Returns:
    (a0, a, b): DC term and first ``N_max'' Fourier coefficients.
  """
  # sawtooth wave at f with amplitude 0.5
  n = np.arange(1, N_max+1)
  a, b = np.zeros(N_max), np.zeros(N_max)
  b[n-1] = (-1)**n * -1 / (np.pi*n)

  # add square wave at 6f (= 2 oct + 5th) with amplitude 0.5
  n_sq = np.arange(1, N_max//6+1, 2)
  b[6*n_sq-1] += 2 / (np.pi * n_sq)

  return (0.0, a, b)
