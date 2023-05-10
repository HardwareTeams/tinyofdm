import numpy as np
import sounddevice as sd
from scipy import signal 
from scipy.signal import resample
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift
import sys

"""
ofdmpy is a super minimal PHY implementation of an OFDM modem
that works in the audio spectrum. thats it. 
"""

class OfdmModem:
    def __init__(self, fft_size, cp_len, fs_bb, fs_tx, fc):
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
        self.fft_size = fft_size # hardcode for now
        self.os_factor = int(self.fs_tx/self.fs_bb)

class TxModem(OfdmModem):
    def __init__(self, fft_size, cp_len, fs_bb, fs_tx, fc):
        OfdmModem.__init__(self, fft_size, cp_len, fs_bb, fs_tx, fc)
    
    def add_cyclic_prefix(self, data, cp=None):
        if(cp is None):
            cp = self.cp_len
        return np.concatenate((data[-cp:], data))

    def gen_symbol(self, data):
        return self.add_cyclic_prefix(ifft(data))
    
    def interpolate_to_fstx(self, data):
        return resample(data, int(len(data)*self.os_factor))
    
    def freq_shift(self, data):
        return np.real(data * np.exp(1j * 2 * np.pi * self.fc/self.fs_tx * np.arange(len(data))))
    
    def transmit(self, data):
        data_u = self.interpolate_to_fstx(data)
        data_fc = self.freq_shift(data_u)


