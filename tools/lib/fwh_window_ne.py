import numpy as np
import os
import sys
if len(sys.argv) != 5:
    #print('Usage: python *.py input_fwh_npy window_length window_ne[0: jump, 1: ne] add_after_window > output_path')
    exit(-1)
input_fwh_path = sys.argv[1]
window_length = int(sys.argv[2])
window_ne = int(sys.argv[3])
add_after_window = int(sys.argv[4])
fwh_data = np.load(input_fwh_path)
input_fwh_name = (input_fwh_path.split('/')[-1]).split('.')[0]
fwh_data = np.reshape(fwh_data, (-1, window_length * 32))
fwh_data_ave = np.zeros((fwh_data.shape[0], 32))
context_half = int((window_length - 1) / 2)  # context_half = 2 if window_length = 5
for i in range(fwh_data.shape[0]):
    if window_ne == 1:
        count = 0
        for j in range(max(i - context_half, 0), min(i + context_half + 1, fwh_data.shape[0])):
            now_part = (i + context_half) - j
            part_start = 32 * now_part
            part_end = 32 * (now_part + 1)
            fwh_data_ave[i] += fwh_data[j][part_start: part_end]
            count += 1
        fwh_data_ave[i] /= count
        #print("fwh_data[", i, "] get average from [", max(i - context_half, 0), ",", min(i + context_half + 1, fwh_data.shape[0]) - 1, "] count is: [", count, "]")
    else:
        pass
if add_after_window > 0:
    sum_ave = np.zeros((fwh_data.shape[0], 32))
    half = int((add_after_window - 1) / 2)
    for i in range(fwh_data.shape[0]):
        if i == 0:
            sum_ave[i] = fwh_data_ave[i]
        else:
            sum_ave[i] = sum_ave[i - 1] + fwh_data_ave[i]
    for i in range(fwh_data.shape[0]):
        if i > half and i < fwh_data.shape[0] - half:
            fwh_data_ave[i] = (sum_ave[i + half] - sum_ave[i - half - 1]) / add_after_window
sys.stdout.buffer.write(fwh_data_ave.astype('float32'))

