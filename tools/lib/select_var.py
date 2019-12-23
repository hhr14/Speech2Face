import sys
import shutil
var_folder = sys.argv[1]
out_path = sys.argv[2]
input_file_path = sys.argv[3]
#  input_file_path example : .../tx_yuhang_020568_e2_fwh32.npy
var_list = ['neutral_var.npy', 'angry_var.npy', 'happy_var.npy', 'sad_var.npy']

filename = (input_file_path.split('/')[-1]).split('.')[0]
emotion = (filename.split('_')[-2])[-1]
shutil.copy(var_folder + var_list[int(emotion)], out_path)
