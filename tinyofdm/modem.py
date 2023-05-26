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

        """
        short training (st) sequence/preamble
        """
        st_seq = np.zeros((self.fft_size), 'complex')
        # ifft in scipy is [0] index is DC
        # 1:nfft/2 is + frequencies
        # nfft/2+1:nfft are - frequencies

        # these are the carriers we will fill for the st 
        sts_idx_pos = self.s2u(np.array([-24, -16, -4, 12, 16, 20, 24]))
        sts_idx_neg = self.s2u(np.array([-20, -12, -8, 4, 8]))
        st_seq[(sts_idx_pos)] = 1+1j
        st_seq[(sts_idx_neg)] = -1-1j
        self.st_ifft_in = st_seq
        self.st_ifft_out = ifft(np.sqrt(13/6)*st_seq)
        sym_temp = self.add_cylic_prefix(self.st_ifft_out)
        self.short_preamble = np.concatenate((sym_temp, sym_temp))

        """
        long training (lt) sequence/preamble
        """
        lt_seq = np.zeros((self.fft_size), 'complex')
        lt_idx = self.s2u(np.arange(-26, 27,1))
        lt_seq[(lt_idx)] = [1, 1, -1, -1, 1,  1, -1, 1,
                           -1, 1,  1,  1, 1,  1,  1,-1, 
                           -1, 1,  1, -1, 1, -1,  1, 1, 
                            1, 1,  0,  1,-1, -1,  1, 1, 
                           -1, 1, -1,  1,-1, -1, -1,-1, 
                           -1, 1,  1, -1,-1,  1, -1, 1, 
                           -1, 1,  1,  1, 1]
        self.lt_ifft_in = lt_seq
        self.lt_ifft_out = ifft(lt_seq)
        self.long_preamble = np.concatenate((self.add_cylic_prefix(self.lt_ifft_out,32), self.lt_ifft_out))


    def s2u(self,list):
        # convert negative fft carriers to positive for scipy
        neg_idxs = np.where(list<0)[0]
        list[neg_idxs] = list[neg_idxs] + self.fft_size
        return list

    def add_cylic_prefix(self,data,cp=None):
        if(cp is None):
            cp = self.cp_len
        return np.concatenate((data[-cp:], data))
    
    def upsample(self, data, I=None):
        if(I is None):
            I = int(self.fs_tx/self.fs_bb)
        temp = np.vstack((data, np.zeros((I-1, len(data)), 'complex')))
        sz = np.shape(temp)
        udata = np.reshape(temp.T, (1,sz[0]*sz[1]))
        return udata[0].T
       
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
    
    def transmit(self, data, audio=False):
        tx = self.freq_shift(self.interpolate_to_fstx(data))
        #play data out of soundcard, also return the data
        #so that we can do loop back tests
        tx /= np.max(np.abs(tx)) # scale data
        if(audio):
            sd.play(np.real(tx), self.fs_tx)
            sd.wait()
        return tx

class RxModem(OfdmModem):
    def __init__(self, fft_size, cp_len, fs_bb, fs_tx, fc):
        OfdmModem.__init__(self, fft_size, cp_len, fs_bb, fs_tx, fc)
    
    def rm_cyclic_prefix(self, data):
       return data[self.cp_len:]
    
    def freq_shift(self, data):
        return data * np.exp(-1j * 2 * np.pi * self.fc/self.fs_tx * np.arange(len(data)))
    
    def decimate(self, data):
        return resample(data, int(len(data)/self.os_factor))
    
    def detect_symbol(self, data, L):
        dout = np.zeros(len(data))
        for ii in range(len(data)-2*L):
            psum = np.dot(data[ii+L:ii+2*L], np.conj(data[ii:ii+L]))
            rsum = np.sum(np.square(np.abs(data[ii+L:ii+2*L])))
            dout[ii] = np.square(np.abs(psum))/np.square(rsum) * (rsum > 0.05)
        idxs = np.where(dout>=0.95)[0]
        return dout, True if idxs.size > 0 else False, idxs[0]
    
    def ofdm_start(self,data):
        L = self.fft_size
        y = np.zeros(len(data)-L)
        for ii in range(len(data)-L):
            y[ii] = np.abs(np.dot(data[ii:ii+L], np.conj(self.lt_ifft_out)))
        indexes_of_detect = np.where(y>=0.85*np.max(y))[0]
        return y, indexes_of_detect[0]-32
    
    def receive(self, data=[], seconds=0., audio=False):
        if(audio):
            rx = np.zeros((int(self.fs_tx*seconds), 1))
            sd.rec(samplerate=self.fs_tx, channel=1, out=rx)
            sd.wait()
            return np.reshape(rx,(len(rx)))
        return self.decimate(self.freq_shift(data))

   

    



