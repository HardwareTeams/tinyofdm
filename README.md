# Project Info

tinyofdm is an OFDM modem implementation that works over audio. 

upper layers of the stack will be built *eventually*. For now tinyofdm is just the physical layer (PHY) of an OFDM modem. 

# Documentation

For full write-ups see: [SignalProcessingJobs.com/docs/audio-ofdm](https://signalprocessingjobs.com/docs/audio-ofdm/)

## What is OFDM? 

OFDM is a method for digital communications using a frequency multiplexing scheme that uses multiple sub-carriers to transmit several symbols of data in parallel. Each sub-carrier can use a common modulation scheme such as quadrature amplitude modulation or phase shift keying. The advantages of OFDM lies in its straight-forward channel equalization scheme, easy elimination of inter-symbol interference, and efficient implementation via the FFT. OFDM is used in WiFI, 5G, digital TV, and a host of other radio communincation.

## Repo 

### Setting up requirements

run:

`pip install -r requirements.txt`

### Running Tests

`python3 -m unittest -v test.test_vis.VisTests.gen_symbol`