import numpy as np 
import unittest
from tinyofdm.modem import TxModem, RxModem
from scipy.fft import fft, fftshift
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

class AudioTests(unittest.TestCase):
   def tx(self):
      tx = TxModem(fft_size, cp, fs_bb, fs_tx, fc)

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
      tx.transmit(tx_all, True)
      assert(True)

   def rx(self):
      rx = RxModem(fft_size, cp, fs_bb, fs_tx, fc)

      # receiver from mic
      rx_raw = rx.receive([], 10.0, True)

      # downconvert
      rx_iq = rx.receive(rx_raw)
      plt.figure
      plt.plot(rx_iq)
      plt.show()
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

      received_data_symbol = rx_iq[rx_data_start:rx_data_start+rx.fft_size]
      received_data = fft(received_data_symbol) * 1/H

      plt.figure
      plt.scatter(np.real(received_data), np.imag(received_data))
      plt.title('iq')
      plt.show()

      print('received data')
      print(np.round(fftshift(received_data),2))