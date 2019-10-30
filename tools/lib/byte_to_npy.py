import numpy as np
import sys
import os
data = np.frombuffer(sys.stdin.buffer.read(), dtype='float32')
input_file_path = sys.argv[1]
input_file_name = input_file_path.split('/')[-1]
output_folder = sys.argv[2]
np.save(os.path.join(output_folder, input_file_name + '.npy'), data)
