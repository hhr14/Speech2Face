import numpy as np
import sys
data = np.load(sys.argv[1])
print(data.shape, data[0].shape)
#print(data[:32])
#print(data[0][0][:64])
