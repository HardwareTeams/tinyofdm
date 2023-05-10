import numpy as np 
import unittest
from ofdmpy.ofdmpy import TxModem
from scipy.fft import fft, ifft, fftshift
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=np.inf)
plt.style.use('dark_background') # i dont want your eyes to hurt
    
fs_bb = 4000.
fs_tx = 48000.
fc    = 10000. 
fft_size = 64
cp = 16
test_data = np.zeros(fft_size, 'complex')
test_data[1:30] = 1

def plot_fft(data, fs, title_str):
    plt.figure()
    xfft = fftshift(20*np.log10(np.abs(fft(data))))
    xaxis = np.arange(-.5*fs, .5*fs, fs/len(data))
    plt.plot(xaxis, xfft)
    plt.xlabel('fs')
    plt.ylabel('dB')
    plt.title(title_str)
    plt.show(block=False)
    plt.pause(0.001)
    input('Press [Enter] to continue...')

class VisTests(unittest.TestCase):
        def gen_symbol(self):
            modem = TxModem(fft_size, cp, fs_bb, fs_tx, fc)
            test_symbol = modem.gen_symbol(test_data)
            plot_fft(test_symbol[16:], fs_bb, 'tx symbol')
            assert(True)

        def interp(self):
            modem = TxModem(fft_size, cp, fs_bb, fs_tx, fc)
            test_symbol = modem.gen_symbol(test_data)
            tx_interp = modem.interpolate_to_fstx(test_symbol[-64:])
            plot_fft(tx_interp, fs_tx, 'tx symbol upsampled')
        
        def freq_shift(self):
            modem = TxModem(fft_size, cp, fs_bb, fs_tx, fc)
            test_symbol = modem.gen_symbol(test_data)
            tx_interp = modem.interpolate_to_fstx(test_symbol[-64:])
            tx_shift = modem.freq_shift(tx_interp)
            plot_fft(tx_shift, fs_tx, 'tx freq. shifted')
        
       