import numpy as np
import sys
mfcc = np.load(sys.argv[1])
assert len(mfcc) % 13 == 0
mfcc = np.reshape(mfcc, (-1, 13))
ppg_dim = int(sys.argv[2])
energy = mfcc[:, -1].reshape((-1, 1))
ppg = np.frombuffer(sys.stdin.buffer.read(), dtype='float32')
assert len(ppg) % ppg_dim == 0
ppg = np.reshape(ppg, (-1, ppg_dim))
energy = energy[:min(energy.shape[0], ppg.shape[0]), :]
ppg = ppg[:min(energy.shape[0], ppg.shape[0]), :]
assert ppg.shape[0] == energy.shape[0]
result_ppg = np.concatenate((ppg, energy), axis=1)
# assert len(result_ppg.reshape(-1)) % 219 == 0
sys.stdout.buffer.write(result_ppg.astype('float32'))
