import sys
import wave
import os
from shutil import copyfile
from pydub import AudioSegment


wav_path = sys.argv[1]
output_folder = sys.argv[2]
split_time = float(sys.argv[3])  # 10.6s
overlap = float(sys.argv[4])
wav_name = (wav_path.split('/')[-1]).split('.')[0]
wav_input = AudioSegment.from_wav(wav_path)
print('duration', wav_input.duration_seconds)
if wav_input.duration_seconds <= split_time:
    copyfile(wav_path, os.path.join(output_folder, wav_name + '.wav'))
    exit(0)
first_segment_length = int((split_time + overlap) * 1000)
inter_segment_length = int((split_time + overlap * 2) * 1000)
overlap_length = int(overlap * 1000)
start = 0
end = first_segment_length
count = 0
while 1:
    split_wav = wav_input[start: end]
    split_wav.export(os.path.join(output_folder, wav_name + '_' + str(count) + '.wav'), format='wav')
    if end == len(wav_input):
        break
    start = end - overlap_length * 3
    end = min(start + inter_segment_length, len(wav_input))
    count += 1

