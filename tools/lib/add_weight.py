import numpy as np
import sys
weight_path = sys.argv[1]
variance 
mlpg_fwh = np.frombuffer(sys.stdin.buffer.read(), dtype='float32')
mlpg_fwh = np.reshape(mlpg_fwh, (-1, 32))
weight_fwh = np.copy(mlpg_fwh)
weight_matrix = []
f = open(weight_path, 'r')
line = f.readline()
while line != '':
    weight_matrix.append(float(line.strip()))
    line = f.readline()
assert len(weight_matrix) == mlpg_fwh.shape[1]
for i in range(len(weight_matrix)):
    weight_fwh[:, i] *= weight_matrix[i]
sys.stdout.buffer.write(weight_fwh.astype('float32'))
