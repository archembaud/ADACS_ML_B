# Filtering Demonstration
# Dr. Matthew Smith, ADACS @ Swinburne University of Technology
# Generate a fake signal, add noise and perform filtering.
# Questions? Email: msmith@astro.swin.edu.au

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

# Create a figure
plt.figure

# Generate time
t = np.linspace(0,0.2,2001)
# Compute genuine signal
x = np.sin(2.0*np.pi*5.0*t) + 0.1*np.sin(2.0*np.pi*50.0*t)
plt.plot(t,x,'k',alpha=0.75)
# Add some noise
x = x+ np.random.randn(len(t))*0.5

# Peform filtering using Butterworth
b,a = signal.butter(3,0.01)
zi = signal.lfilter_zi(b,a)
z, _ = signal.lfilter(b,a,x, zi=zi*x[0])
filt_x = signal.filtfilt(b,a,x)

# Plot
plt.plot(t,x,'b',alpha=0.75)
plt.plot(t,filt_x,'r')
plt.title('Comparison of filtered result and original signal')
plt.xlabel('Time, s')
plt.ylabel('Signal, x')
plt.legend(('Real Signal','Signal+Noise','Filtered Result'))
plt.show()
