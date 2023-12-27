import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
from scipy import signal
import scipy.io.wavfile
data = pd.read_csv('TREMOR12Data.csv')

x = data["timestamp2001_ms"]

accx, accy, accz = data['accX'], data['accY'], data['accZ'] # accelerometer data
rotx, roty, rotz = data['rotX'], data['rotY'], data['rotZ'] # rotation speed data

x = round((x - x[0]) / 1000, 2)

figure, axis = plt.subplots(2, 1)


# Accelerometer

axis[0].plot(x, accx, 'r', label="X")
axis[0].plot(x, accy, 'b', label="Y")
axis[0].plot(x, accz, 'g', label="Z")
axis[0].set_title("Accelerometer Data")
axis[0].set_xlabel("Time (s)")
axis[0].set_ylabel("Acceleration (g)")
axis[0].legend(frameon=False)



def bandpass_filter(sig):
    fs = 100 # each data point is sampled over 10 ms, getting a fs of 100
    lowcut = 3
    highcut = 7

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    order = 2

    b, a = signal.butter(order, [low, high], 'bandpass', analog=False)
    y = signal.filtfilt(b, a, sig, axis=0)

    return (y)


fx, fy, fz = bandpass_filter(accx), bandpass_filter(accy), bandpass_filter(accz)

axis[1].plot(x, fx, 'r', label="X")
axis[1].plot(x, fy, 'b', label="Y")
axis[1].plot(x, fz, 'g', label="Z")
axis[1].set_title("Filtered Accelerometer Data")
axis[1].set_xlabel("Time (s)")
axis[1].set_ylabel("Acceleration (g)")
axis[1].legend(frameon=False)
plt.show()
