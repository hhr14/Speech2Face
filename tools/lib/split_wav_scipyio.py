import sys
import wave
import os
from shutil import copyfile
from scipy.io import wavfile


wav_path = sys.argv[1]
output_folder = sys.argv[2]
split_time = float(sys.argv[3])  # 10.6s
overlap = float(sys.argv[4])  # 1s
wav_name = (wav_path.split('/')[-1]).split('.')[0]
frame_rate, raw_wave = wavfile.read(wav_path)
frame_number = len(raw_wave)
print("frame_rate", frame_rate, "frame_number", frame_number, "duration", frame_number / frame_rate)
if frame_rate * split_time >= frame_number:
    copyfile(wav_path, os.path.join(output_folder, wav_name + '.wav'))
    exit(0)
first_segment_length = int((split_time + overlap) * frame_rate)
inter_segment_length = int((split_time + overlap * 2) * frame_rate)
overlap_segment_length = int(overlap * frame_rate)
start_frame = 0
end_frame = first_segment_length
count = 0
while 1:
    wavfile.write(os.path.join(output_folder, wav_name + '_' + str(count) + '.wav'), frame_rate, raw_wave[start_frame: end_frame])
    print("split_wave_length", count, (end_frame - start_frame) / frame_rate)
    if end_frame == frame_number:
        break
    start_frame = end_frame - 3 * overlap_segment_length 
    end_frame = min(start_frame + inter_segment_length, frame_number)
    count += 1
