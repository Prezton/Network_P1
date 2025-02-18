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
    # length => output => symbol => bits2 => bits1 => message

    nfft = 64
    Interleave = np.reshape(np.transpose(np.reshape(np.arange(1, 2*nfft+1, 1),[-1,4])),[-1,])
    preamble = np.array([1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1,1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1])

    if level >= 4:
        mod = comm.modulation.QAMModem(4)
        start_sign = mod.modulate(preamble.astype(bool))
        start_sign = np.fft.ifft(start_sign)
        max_correlatd_value = 0

        correlation_array = np.absolute(np.correlate(output, start_sign))
        for i in range(len(correlation_array)):
            if max_correlatd_value < correlation_array[i]:
                max_correlatd_value = correlation_array[i]
                noise_pad_begin_length = i
        output = output[noise_pad_begin_length:]
        print("decoded starting noise length: ", noise_pad_begin_length)
        begin_zero_padding = noise_pad_begin_length

    if level >= 3:
        nsym = int(len(output)/nfft)
        for i in range(nsym):
            symbol = output[i*nfft:(i+1)*nfft]
            output[i*nfft:(i+1)*nfft] = np.fft.fft(symbol)

    if level >= 2:

        # Remove Preamble
        tmp_message = np.split(output, [len(preamble) // 2])[1]
        [encoded_length, encoded_message] = np.split(tmp_message, [nfft])


        mod = comm.modulation.QAMModem(4)
        decoded_length = mod.demodulate(encoded_length, 'hard')
        length_binary = decoded_length.astype(np.int8)
        length_character = [str(c) for c in length_binary]
        length_character = "".join(length_character)
        length = int(length_character, 2)
        print("decoded message length: ", length)

        # Soft Viterbi Decoding
        # Split out ending noises, * 8 * 2 is from unpack() int lvl1 (1->8 bits) and conv_encodin (1->2 bits)
        decoded_message = softViterbiDecode(encoded_message[:length * 8 * 2])

        output = np.concatenate((np.array(decoded_length, dtype=np.int8), np.array(decoded_message, dtype=np.uint8)))


    if level >= 1:
        [length_binary, output_binary] = np.split(output, [2 * nfft])
        length_binary = length_binary.astype(np.int8)
        length_character = [str(c) for c in length_binary]
        length_character = "".join(length_character)

        # reverse of binary_repr
        length = int(length_character, 2)

        output_binary = output_binary.astype(np.int8)

        # num of symbols
        # chunks of packets
        nsym = int(len(output_binary) / (2 * nfft))
        for i in range(nsym):
            tmp = output_binary[i*2*nfft:(i+1)*2*nfft]
            tmp = np.reshape(tmp, [4, -1])
            # arange->reshape->transpose->reshape
            tmp = np.transpose(tmp)
            tmp = np.reshape(tmp, [-1, ])
            output_binary[i*2*nfft:(i+1)*2*nfft] = tmp
        message_binary = np.split(output_binary, [8 * length])[0]
        message_binary = np.packbits(message_binary.astype(np.int8))
        message = [chr(c) for c in message_binary]
        message = "".join(message)

    return begin_zero_padding, message, length

def softViterbiDecode(message):

    trellis = {
        # current state: { previous state : [input i(n), output c(n)]}
        0: {0: [0, complex(-1, -1)], 2: [0, complex(1, 1)]},
        1: {0: [1, complex(1, 1)], 2: [1, complex(-1, -1)]},
        2: {1: [0, complex(1, -1)], 3: [0, complex(-1, 1)]},
        3: {1: [1, complex(-1, 1)], 3: [1, complex(1, -1)]}
    }

    # node_metric represents the shortest distance from beginning to this node
    node_metric = {0: 0, 1: float("inf"), 2: float("inf"), 3: float("inf")}

    # path is used to mark the best routes for backtrace
    path = {0: [], 1: [], 2: [], 3: []}
    for symbol in message:
        for current_state in trellis:
            prev_state_1 = list(trellis[current_state].keys())[0]
            prev_state_2 = list(trellis[current_state].keys())[1]
            distance_1 = node_metric[prev_state_1] + np.linalg.norm(trellis[current_state][prev_state_1][1] - symbol)
            distance_2 = node_metric[prev_state_2] + np.linalg.norm(trellis[current_state][prev_state_2][1] - symbol)
            # Find the shortest path accordingly
            if distance_1 < distance_2:
                # update shortest path value so far
                node_metric[current_state] = distance_1
                path[current_state].append(prev_state_1)
            else:
                node_metric[current_state] = distance_2
                path[current_state].append(prev_state_2)

    min_metric = float("inf")
    # Find the end point
    for node in node_metric:
        if node_metric[node] < min_metric:
            min_node = node
            min_metric = node_metric[node]

    idx = -1
    decoded_result = []
    decoded_length = len(message)
    # Backtrace
    for i in range(decoded_length):
        prev_branch = path[min_node][idx]
        decoded_result.append(trellis[min_node][prev_branch][0])
        min_node = prev_branch
        idx -= 1
    decoded_result = decoded_result[::-1]

    return np.asarray(decoded_result)







if __name__ == "__main__":
    # txsignal = wifitransmitter.WifiTransmitter('abcdefg', 4)
    # print(txsignal)
    padding_length, txsignal, total_length = wifitransmitter.WifiTransmitter('hello world', 4, 6)
    print("input message length: ", total_length)
    print("actual noise length: ", padding_length)
    print(WifiReceiver(txsignal, 4))
    # softViterbiDecode(np.array("1101100100010111"))