# Documentation

For full write-ups see: [SignalProcessingJobs.com/docs/audio-ofdm](https://signalprocessingjobs.com/docs/audio-ofdm/)

# Project Info

An OFDM modem implementation that works over audio. Maybe we will build the upper layers of the stack *eventually*. For now we are going to be building the physical (PHY) layer of an OFDM modem. OFDM is used in WiFI, 5G, digital TV, and a host of other radio communincation.

## What is OFDM? 

OFDM is a method for digital communications using a frequency multiplexing scheme that uses multiple sub-carriers to transmit several symbols of data in parallel. Each sub-carrier can use a common modulation scheme such as quadrature amplitude modulation or phase shift keying. The advantages of OFDM lies in its straight-forward channel equalization scheme, easy elimination of inter-symbol interference, and efficient implementation via the FFT