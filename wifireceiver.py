#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import sys
import commpy as comm
import commpy.channelcoding.convcode as check
import wifitransmitter

def WifiReceiver(output, level):
    # for lvl 1-3, num of zero padding is 0
    begin_zero_padding = 0

    message = "hello world"
    # length => output => symbol => bits2 => bits1 => message

    nfft = 64
    Interleave = np.reshape(np.transpose(np.reshape(np.arange(1, 2*nfft+1, 1),[-1,4])),[-1,])
    if (level >= 1):
        [length_binary, output_binary] = np.split(output, [2 * nfft])
        length_binary = length_binary.astype(np.int8)
        length_character = [str(c) for c in length_binary]
        length_character = "".join(length_character)

        # reverse of binary_repr
        length = int(length_character, 2)
        # print("received message length: ", length)
        output_binary = output_binary.astype(np.int8)
        # print(np.shape(output_binary))

        # num of symbols
        # chunks of packets
        nsym = int(len(output_binary) / (2 * nfft))
        for i in range(nsym):
            tmp = output_binary[i*2*nfft:(i+1)*2*nfft]
            tmp = np.reshape(tmp, [4, -1])
            # print("tmp shape: ", np.shape(tmp))
            # print("output binary shape: ", np.shape(output_binary))

            # arange->reshape->transpose->reshape
            tmp = np.transpose(tmp)
            tmp = np.reshape(tmp, [-1, ])
            output_binary[i*2*nfft:(i+1)*2*nfft] = tmp
        message_binary = np.split(output_binary, [8 * length])[0]
        message_binary = np.packbits(message_binary.astype(np.int8))
        message = [chr(c) for c in message_binary]
        message = "".join(message)
        # print(message)



    return begin_zero_padding, message, length

if __name__ == "__main__":
    txsignal = wifitransmitter.WifiTransmitter('ABC', 1)
    # print(txsignal)
    print("input length: ", len('ABC'))
    print(WifiReceiver(txsignal, 1))