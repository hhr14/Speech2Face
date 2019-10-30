import sys
import os
import numpy as np


def interpolate(seq1, seq2):
    time_length = len(seq1)
    inter_seq = np.zeros(seq1.shape)
    print("seq1.shape", seq1.shape)
    for i in range(time_length):
        weight = i / (time_length - 1)
        inter_seq[i] = (1 - weight) * seq1[i] + weight * seq2[i]
    for i in range(time_length):
        print("seq1 ", seq1[i][0], "seq2 ", seq2[i][0], "inter_seq ", inter_seq[i][0])
    return inter_seq


ppg_folder = sys.argv[1]
ppg_output_folder = sys.argv[2]
overlap = float(sys.argv[3])
ppg_frame_rate = float(sys.argv[4])  # 100fps
overlap_frame = int(overlap * ppg_frame_rate)
file_list = os.listdir(ppg_folder)
file_list.sort(key=lambda ppg_file: int((ppg_file.split('_')[-1]).split('.')[0]))
ppg_data = []
ppg_name = []
count = 0
print('ppg after wav2ppg')
for ppg_file in file_list:
    temp = np.load(os.path.join(ppg_folder, ppg_file))
    ppg_data.append(np.reshape(temp, (-1, 218)))
    ppg_name.append(ppg_file)
    print(ppg_file, str(count), ppg_data[-1].shape)
    count += 1
for i in range(len(ppg_data)):
    if i >= 1:
        ppg_data[i] = ppg_data[i][overlap_frame:]
        interpolate_frame = interpolate(ppg_data[i - 1][-overlap_frame:], ppg_data[i][:overlap_frame])
        ppg_data[i-1][-overlap_frame:] = interpolate_frame
        ppg_data[i][:overlap_frame] = interpolate_frame
    if i < len(ppg_data) - 1:
        ppg_data[i] = ppg_data[i][:-overlap_frame]

#  test 
print('ppg after interpolate')
for i in range(len(ppg_data)):
    print(str(i), ppg_data[i].shape)
    np.save(os.path.join(ppg_output_folder, ppg_name[i]), ppg_data[i])

