import sys
import os
folder_path = sys.argv[1]
file_list = os.listdir(folder_path)
file_prefix = ''
wave_name = ''
for myfile in file_list:
    file_prefix = myfile[:-11]  #  format: xxx_e0 delete(_xxxxxx.png)
    wave_name = file_prefix
    break
final_prefix = os.path.join(folder_path, file_prefix) + '_%06d.png ' + file_prefix + '.mp4 ' +  wave_name + '.wav'
sys.stdout.buffer.write(str.encode(final_prefix))
