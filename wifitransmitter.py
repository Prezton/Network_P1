#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import sys
import commpy as comm
import commpy.channelcoding.convcode as check

def WifiTransmitter(*args):
    # Default Values
    if len(args)<2:
        # Arg1 = Message, Arg2 = Level, Arg3 = SNR
        message = args[0]
        level=4
        snr=np.Inf
    elif len(args)<3:
        # Arg1 = Message, Arg2 = Level, Arg3 = SNR
        message=args[0]
        level=int(args[1])
        snr=np.Inf
    elif len(args)<4:
        # Arg1 = Message, Arg2 = Level, Arg3 = SNR
        message=args[0]
        level=int(args[1])
        snr=int(args[2])
    
	## Sanity checks
    if len(message) > 10000:
        raise Exception("Error: Message is too long")
    if level>4 or level<1:
        raise Exception("Error:Invalid Level, must be 1-4")


    nfft = 64
    Interleave = np.reshape(np.transpose(np.reshape(np.arange(1, 2*nfft+1, 1),[-1,4])),[-1,])
    # arange->reshape->transpose->reshape
    # Interleave: n * 4 --> 4 * n, then reshape to 1 * 4n
    # print("Interleave: ", Interleave)
    length = len(message)
    preamble = np.array([1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1,1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1])
    cc1 = check.Trellis(np.array([3]),np.array([[0o7,0o5]]))
    if level >= 1:
        bits = np.unpackbits(np.array([ord(c) for c in message], dtype=np.uint8))
        # bits: binary represent of chars, padding 0 at the end
        # print("bits_1 length: ", len(bits))

        bits = np.pad(bits, (0, 2*nfft-len(bits)%(2*nfft)),'constant')
        # print("bits_2 length: ", len(bits))
        # len(bits) == 2 * nfft

        nsym = int(len(bits)/(2*nfft))
        # num of symbols
        # chunks of packets
        output = np.zeros(shape=(len(bits),))
        for i in range(nsym):
            symbol = bits[i*2*nfft:(i+1)*2*nfft]
            output[i*2*nfft:(i+1)*2*nfft] = symbol[Interleave-1]
        # print("symbol is: ", symbol)
        # print("output_1 is: ", output)
        len_binary = np.array(list(np.binary_repr(length).zfill(2*nfft))).astype(np.int8)
        # print("len_binary is: ", len_binary)
        output = np.concatenate((len_binary, output))
        # print("output is: ", output)
        # print("len_binary length: output length: ", len(len_binary), len(output))

    if level >= 2:
        coded_message = check.conv_encode(output[2*nfft:].astype(bool), cc1)
        # convolutional encoding every bits except for length bits
        coded_message = coded_message[:-6]
        # why [:-6]?
        output = np.concatenate((output[:2*nfft],coded_message))
        # concatenate with length
        # print("output before QAM modulation and preamble is: ", output, "length is: ", len(output))
        output = np.concatenate((preamble, output))
        # concatenate with preamble
        mod = comm.modulation.QAMModem(4)
        # QAM Modulation
        output = mod.modulate(output.astype(bool))
        
    if level >= 3:
        nsym = int(len(output)/nfft)
        for i in range(nsym):
            symbol = output[i*nfft:(i+1)*nfft]
            output[i*nfft:(i+1)*nfft] = np.fft.ifft(symbol)
    
    if level >= 4:
        noise_pad_begin = np.zeros(np.random.randint(1,1000))
        noise_pad_begin_length = len(noise_pad_begin)
        noise_pad_end = np.zeros(np.random.randint(1,1000))
        output = np.concatenate((noise_pad_begin,output,noise_pad_end))        
        output = comm.channels.awgn(output,snr)
        return noise_pad_begin_length, output, length
            
    return output
    
if __name__ == '__main__':
    if len(sys.argv)<2:
        raise Exception("Error: No message was provided")
    elif len(sys.argv)<3:
        WifiTransmitter(sys.argv[1])
    elif len(sys.argv)<4:
        WifiTransmitter(sys.argv[1], sys.argv[2])
    elif len(sys.argv)<5:
        WifiTransmitter(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv)>=5:
        raise Exception("Error: Number of arguments exceed the maximum arguments allowed (3)")