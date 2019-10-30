import sys
import os
import numpy as np


def interpolate(seq1, seq2):
    time_length = len(seq1)
    inter_seq = np.zeros(seq1.shape)
    for i in range(time_length):
        weight = i / time_length
        inter_seq[i] = (1 - weight) * seq1[i] + weight * seq2[i]
    return inter_seq


if len(sys.argv) != 6:
    print("Usage python *.py split_fwh_folder concat_fwh_folder overlap[1] fwh_frame_rate[100] feature_dim[96]")
    exit(-1)


fwh_folder = sys.argv[1]
fwh_output_folder = sys.argv[2]
overlap = float(sys.argv[3])  # 1
fwh_frame_rate = float(sys.argv[4])  # 100fps
feature_dim = int(sys.argv[5])  # 96
overlap_frame = int(overlap * fwh_frame_rate)
file_list = os.listdir(fwh_folder)
file_list.sort(key=lambda fwh_file: int((fwh_file.split('_')[-3])))
fwh_data = []
fwh_name = '_'.join(file_list[0].split('_')[:-3]) + '_' + file_list[0].split('_')[-2]
count = 0
fwh_seq = []
for fwh_file in file_list:
    temp = np.load(os.path.join(fwh_folder, fwh_file))
    fwh_data.append(np.reshape(temp, (-1, feature_dim)))
    print(fwh_file, str(count), fwh_data[-1].shape)
    count += 1
for i in range(len(fwh_data)):
    if i == 0:
        fwh_seq.extend(fwh_data[i][:-overlap_frame])
    elif i == len(fwh_data) - 1:
        interpolate_frame = interpolate(fwh_data[i-1][-overlap_frame:], fwh_data[i][:overlap_frame])
        fwh_seq.extend(interpolate_frame)
        print("inter range: ", (len(fwh_seq) - overlap_frame) / fwh_frame_rate, len(fwh_seq) / fwh_frame_rate)
        fwh_seq.extend(fwh_data[i][overlap_frame:])
    else:
        interpolate_frame = interpolate(fwh_data[i-1][-overlap_frame:], fwh_data[i][:overlap_frame])
        fwh_seq.extend(interpolate_frame)
        print("inter range: ", (len(fwh_seq) - overlap_frame) / fwh_frame_rate, len(fwh_seq) / fwh_frame_rate)
        fwh_seq.extend(fwh_data[i][overlap_frame:-overlap_frame])
    print(i, 'fwh_seq_length', len(fwh_seq))


#  test 
print('fwh after interpolate')
print("fwh_seq.shape", np.array(fwh_seq).shape)
np.save(os.path.join(fwh_output_folder, fwh_name), fwh_seq)

