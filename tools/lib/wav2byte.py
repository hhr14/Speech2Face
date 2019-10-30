from scipy.io import wavfile
import sys
import numpy as np
wav_path = sys.argv[1]
frame_rate, raw_wave = wavfile.read(wav_path)
frame_number = len(raw_wave)
sys.stdout.buffer.write(np.array(raw_wave).astype('float32'))
