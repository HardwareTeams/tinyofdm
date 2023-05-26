import numpy as np 
import unittest
from tinyofdm.modem import TxModem, RxModem
from scipy.fft import fft, ifft, fftshift
from scipy.signal import lfilter
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
def plot_time(data, title_str):
    plt.figure()
    plt.plot(data)
    plt.xlabel('sample')
    plt.ylabel('mag')
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

class LoopBack(unittest.TestCase):
     def loop_back(self):
        test_chan = np.array([.8, .0001, .0001, .001, .1+.1j])
        tx = TxModem(fft_size, cp, fs_bb, fs_tx, fc)
        rx = RxModem(fft_size, cp, fs_bb, fs_tx, fc)

        bpsk_seq = np.zeros(tx.fft_size, 'complex')
        bpsk_idx = tx.s2u(np.arange(-26, 27, 1))

        bpsk_seq[(bpsk_idx)] = [-.7-.7j,.7+.7j, .7-.7j,-.7+.7j, 1, 1, 1, 1,
                                -1, 1, 1,  1,  1,  1,  1, -1, 
                                -1, 1, 1, -1,  1, -1,  1,  1, 
                                 1, 1, 0,  1, -1, -1,  1,  1, 
                                -1, 1, -1, 1, -1, -1, -1, -1, 
                                -1, 1,  1,-1, -1,  1, -1,  1, 
                                -1, 1, .4, -1,    .5+.5j]

        bpsk_test_seq = tx.gen_symbol(bpsk_seq)

        tx_all = np.concatenate((np.zeros(500,'complex'),\
                                tx.short_preamble,\
                                tx.long_preamble,\
                                bpsk_test_seq,\
                                np.zeros(500, 'complex')))

        # transmit data, passthrough
        tx_data = tx.transmit(tx_all, False)

        # emulate channel
        rx_raw = lfilter(tx.upsample(test_chan, tx.os_factor), [1.], tx_data) + \
                 np.random.normal(0, 0.05, len(tx_data))
        
        # downconvert
        rx_iq = rx.receive(rx_raw)

        #detection chain
        rx_st_det, found_start, rx_start_st = rx.detect_symbol(rx_iq, 16)
        rx_lt_det, rx_start_lt = rx.ofdm_start(rx_iq)

        # for now calculating offsets, todo:cleanup
        rx_start_lt += 32
        rx_data_start = rx_start_lt + (160-32) + 16
        lt_received = rx_iq[rx_start_lt:rx_start_lt+128]

        # calculate channel response
        LTS0 = fft(lt_received[:rx.fft_size])
        LTS1 = fft(lt_received[-rx.fft_size:])
        H = (0.5 * (LTS0+LTS1))/rx.lt_ifft_in

        plt.figure
        plt.plot(fftshift(20*np.log10(np.abs(1/H))))
        plt.plot(fftshift(20*np.log10(np.abs(fft(test_chan,64)))))
        plt.title('channel estimation')
        plt.show()

        received_data_symbol = rx_iq[rx_data_start:rx_data_start+rx.fft_size]
        received_data = fft(received_data_symbol) * 1/H

        plt.figure
        plt.scatter(np.real(received_data), np.imag(received_data))
        plt.title('iq')
        plt.show()

        print('received data')
        print(np.round(fftshift(received_data),2))
        assert(True)

       