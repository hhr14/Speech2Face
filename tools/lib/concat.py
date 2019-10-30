import numpy as np
import sys
dim = int(sys.argv[1])
mfcc = np.load(sys.argv[2]).reshape((-1, dim))
ppg = np.frombuffer(sys.stdin.buffer.read(), dtype='float32').reshape((-1, 218))
mfcc = mfcc[:ppg.shape[0], :]
result_ppg = np.concatenate((ppg, mfcc), axis=1)
sys.stdout.buffer.write(result_ppg.astype('float32'))
