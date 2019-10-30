import numpy as np
import sys
file_path = sys.argv[1]
feature = np.load(file_path)
sys.stdout.buffer.write(feature.astype('float32'))
