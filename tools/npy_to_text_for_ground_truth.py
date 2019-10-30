import numpy as np
import sys
import os
out_folder = sys.argv[2]
input_file_path = sys.argv[1]
input_file_name = (input_file_path.split('/')[-1]).split('.')[0]
data = np.load(input_file_path)
data = np.reshape(data, (-1, 32))
f = open(os.path.join(out_folder, input_file_name + '_gourndtruth.txt'), 'w')
for i in range(data.shape[0]):
    out = ''
    for j in range(data.shape[1]):
        out += str(data[i][j]) + ' '
    out += '\n'
    f.write(out)

