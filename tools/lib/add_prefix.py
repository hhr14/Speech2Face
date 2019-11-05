import numpy as np
import sys
import os

if len(sys.argv) != 3:
    print('Usage: python *.py folder_path prefix[tx_yuhang_/bb_f15_]')
    exit(1)

folder = sys.argv[1]
prefix = sys.argv[2]
file_list = os.listdir(folder)
for file in file_list:
    old_file_name = os.path.join(folder, file)
    new_file_name = os.path.join(folder, prefix + file)
    os.rename(old_file_name, new_file_name)
