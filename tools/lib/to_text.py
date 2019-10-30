import numpy as np
import sys
import os
data_ = np.frombuffer(sys.stdin.buffer.read(), dtype='float32')
data = data_.copy()
data = np.reshape(data, (-1, 32))
out_folder = sys.argv[1]
input_file_path = sys.argv[2]
set_30e0_0e1_31e500 = sys.argv[3]
input_file_name = (input_file_path.split('/')[-1]).split('.')[0]
f = open(os.path.join(out_folder, input_file_name + '.txt'), 'w')
if set_30e0_0e1_31e500 == "1":
    data[:, 0] = 1.0
    data[:, 30] = 0.0
    data[:, 31] = 500.0
for i in range(data.shape[0]):
    out = ''
    for j in range(data.shape[1]):
        out += str(data[i][j]) + ' '
    out += '\n'
    f.write(out)

