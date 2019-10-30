#!/usr/bin/env python3

import sys
import numpy as np
from scipy.interpolate import interp1d
import os



if __name__ == '__main__':
    if len(sys.argv) != 7:
        print("Usage python emotion[0:normal 1:angry 2:happy 3: sad] input_file_path output_file_path from_byte[0:False 1:True] from_name[0:False, 1:True] ppg_dim  < input_file_path")
        exit(-1)
    emotion = sys.argv[1]
    input_file_path = sys.argv[2]
    output_file_path = sys.argv[3]
    from_byte = int(sys.argv[4])  # 0 is from byte, 1 is from input_file
    from_name = int(sys.argv[5])
    dim = int(sys.argv[6])
    vector = [0.0, 0.0, 0.0, 0.0]
    filename = (input_file_path.split('/')[-1]).split('.')[0]
    if from_name == 1:
        index = filename.split('_')[-1]
        if index[:2] == "01":
            emotion = "1"
        elif index[:2] == "02":
            emotion = "2"
        elif index[:2] == "03":
            emotion = "3"
        else:
            emotion = "0"
    if emotion == "1":
        vector[1] = 1.0
    elif emotion == "2":
        vector[2] = 1.0
    elif emotion == "3":
        vector[3] = 1.0
    else:
        vector[0] = 1.0
    #print(vector)
    if from_byte == 1:
        ppg = np.frombuffer(sys.stdin.buffer.read(),dtype='float32')
    else:
        ppg = np.load(input_file_path)
    ppg = np.reshape(ppg, (-1, dim))
    vector = np.tile(vector, (ppg.shape[0], 1))
    result = np.concatenate((ppg, vector), axis=1)
    #x = np.arange(0,y.shape[0])/sr_from
    #duration = y.shape[0]/sr_from
    #x2= np.arange(0,duration,1/sr_to)

    #f = interp1d(x,y,kind='quadratic',axis=0,fill_value='extrapolate',copy=False,assume_sorted=True)
    #print(result.shape) 
    #print(result)
    np.save(os.path.join(output_file_path, filename + '_e' + emotion), result)
    #exit(0)

