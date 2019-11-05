import numpy as np
import sys
a = np.frombuffer(sys.stdin.buffer.read(), dtype='float32')
dim = int(sys.argv[1])
assert a[0] == 0 and a[1] == 0 and a[2] == 0
res = a[5:]
assert len(res) % dim == 0
sys.stdout.buffer.write(res.astype('float32'))
