import numpy as np
import sys
dim = int(sys.argv[1])
speaker_id = int(sys.argv[2])  # from 0 to 2: 0 yuhang 1 ff15 2 m10
data = np.frombuffer(sys.stdin.buffer.read(), dtype='float32')
if len(data) % dim != 0:
    np.save('1', data)
assert len(data) % dim == 0
data = np.reshape(data, (-1, dim))
speaker_embedding = np.zeros(3)
speaker_embedding[speaker_id] = 1
speaker_matrix = np.tile(speaker_embedding, (data.shape[0], 1))
res = np.concatenate((data, speaker_matrix), axis=1)
sys.stdout.buffer.write(res.astype('float32'))
