import numpy as np
import sounddevice as sd
from scipy import signal 
from scipy.signal import resample
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift
import sys
plt.style.use('dark_background') # i dont want your eyes to hurt

class ofdmpy:
    """
    ofdmpy is a super minimal PHY implementation of an OFDM modem
    that works in the audio spectrum. thats it. 
    """

    def __init__(self, cp_len, fs_bb, fs_tx, fc):
        """
        cp_len = cyclic prefix length
        fs_bb = baseband sampling rate
        fs_tx = transmit sampling rate
        fc = transmit center frequency
        """
        self.cp_len = cp_len
        self.fs_bb = fs_bb
        self.fs_tx = fs_tx
        self.fc = fc 
        self.fft_size = 64 # hardcode for now
        self.os_factor = int(self.fs_tx/self.fs_bb)

    def add_cyclic_prefix(self, data, cp=None):
        if(cp is None):
            cp = self.cp_len
        return np.concatenate((data[-cp:], data))

    def gen_symbol(self, data):
        return self.add_cyclic_prefix(ifft(data))
    
    def interpolate_to_fstx(self, data):
        return resample(data, int(len(data)*self.os_factor))
    
    def freq_shift(self, data):
        samples = np.arange(len(data))
        # we take the real part because our data will be complex
        # but we cant transmit a complex signal 
        return np.real(data * np.exp(1j * 2 * np.pi * self.fc/self.fs_tx * samples))
    
    def transmit(self, data):
        data_t0 = self.interpolate_to_fstx(data)
        data_t1 = self.freq_shift(data_t0)
        # more to do 

    # lets build a test to make sure things to work so far

    def test(self):
        tx_data = np.zeros(64, 'complex')
        tx_data[1:30] = 1 # fill 1/2 the ifft up with 1's
        tx_data_time = self.gen_symbol(tx_data)

        plt.figure()
        xfft = fftshift(20*np.log10(np.abs(fft(tx_data_time[-64:]))))
        xaxis = np.arange(-.5*self.fs_bb, .5*self.fs_bb, self.fs_bb/self.fft_size)
        plt.plot(xaxis, xfft)
        plt.xlabel('fs_bb')
        plt.ylabel('dB')
        plt.title('transmit symbol frequency domain')
        plt.show()

        tx_interp = self.interpolate_to_fstx(tx_data_time[-64:])
        plt.figure()
        xfft = fftshift(20*np.log10(np.abs(fft(tx_interp))))
        xaxis = np.arange(-.5*self.fs_tx, .5*self.fs_tx, self.fs_tx/len(tx_interp))
        plt.plot(xaxis, xfft)
        plt.xlabel('fs_tx')
        plt.ylabel('dB')
        plt.title('transmit (upsampled) symbol frequency domain')
        plt.show()

        tx_shift = self.freq_shift(tx_interp)
        plt.figure()
        xfft = fftshift(20*np.log10(np.abs(fft(tx_shift))))
        xaxis = np.arange(-.5*self.fs_tx, .5*self.fs_tx, self.fs_tx/len(tx_shift))
        plt.plot(xaxis, xfft)
        plt.xlabel('fs_tx')
        plt.ylabel('dB')
        plt.title('transmit (freq shifted) symbol frequency domain')
        plt.show()

o = ofdmpy(16, 4000., 48000., 10000.)
o.test()