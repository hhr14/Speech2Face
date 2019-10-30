import numpy as np
import sys
data = np.frombuffer(sys.stdin.buffer.read(), dtype='float32')
print(data.shape)
