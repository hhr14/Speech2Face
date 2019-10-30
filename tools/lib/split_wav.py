import sys
import wave
import os
from shutil import copyfile


def write_wave(wav_input, wav_segment, output_path):
    wav_output = wave.open(output_path, 'wb')
    #wav_output.setparams(wav_input.getparams())
    print("len_wav_segment", len(wav_segment))
    print("time", len(wav_segment) / wav_input.getparams().framerate)
    wav_output.setnframes(len(wav_segment))
    wav_output.setnchannels(wav_input.getnchannels())
    wav_output.setframerate(wav_input.getframerate())
    wav_output.setsampwidth(wav_input.getsampwidth())
    wav_output.writeframes(wav_segment)
    wav_output.close()


wav_path = sys.argv[1]
output_folder = sys.argv[2]
split_time = float(sys.argv[3])  # 10.6s
overlap = float(sys.argv[4])  # 1s
wav_name = (wav_path.split('/')[-1]).split('.')[0]
wav_input = wave.open(wav_path, 'rb')
frame_number = wav_input.getparams().nframes
frame_rate = wav_input.getparams().framerate
print("channels", wav_input.getparams().nchannels)
if frame_rate * split_time >= frame_number:
    copyfile(wav_path, os.path.join(output_folder, wav_name + '.wav'))
    exit(0)
raw_wave = wav_input.readframes(frame_number)
first_segment_length = int((split_time + overlap) * frame_rate)
inter_segment_length = int((split_time + overlap * 2) * frame_rate)
overlap_segment_length = int(overlap * frame_rate)
start_frame = 0
end_frame = first_segment_length
count = 0
while 1:
    write_wave(wav_input, raw_wave[start_frame: end_frame], os.path.join(output_folder, wav_name + '_' + str(count) + '.wav'))
    if end_frame == frame_number:
        break
    start_frame = end_frame - 3 * overlap_segment_length 
    end_frame = min(start_frame + inter_segment_length, frame_number)
    count += 1
