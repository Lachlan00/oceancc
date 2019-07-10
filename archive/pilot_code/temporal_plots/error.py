# reporduce issue
import numpy as np
from wavelets import WaveletAnalysis
import matplotlib.pyplot as plt
import matplotlib

# normalisation function
def normalize(data, a, b):
    return [(b-a)*(((x-data.min())/(data.max()-data.min())))+a for x in data]

# create example data
np.random.seed(420)
x = np.random.rand(23)
time = np.asarray(range(1994,2017,1))
# Wavelet analysis
dt = 1
wa = WaveletAnalysis(x, time=time, dt=dt)
# wavelet power spectrum
power = wa.wavelet_power
# normalise (unitless metric)
power = normalize(power, -5, 5)
# scales 
scales = wa.scales
# associated time vector
t = wa.time

# plot
fig1, ax1 = plt.subplots(figsize=(12,3))
T, S = np.meshgrid(t, scales)
plt.contourf(T, S, power, list(range(-5, 6, 1)), cmap='plasma')
cbar = plt.colorbar(ticks=[-5, 0, 5])
cbar.set_label('Wavelet Power Spectrum', rotation=270, labelpad=16)
ax1.set_yscale('log')
ax1.set_ylim(2,8) # This is line is what is causing errors...
ax1.set_yticks([2, 4, 8])
ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax1.set_ylabel('Log Transformed Period (years)', labelpad=16)
plt.xticks(np.arange(1994, 2017, step=2))
plt.gca().invert_yaxis()
# line CI
C, S2 = wa.coi
plt.plot(C, S2, '--', color='grey', alpha=1)

plt.show()