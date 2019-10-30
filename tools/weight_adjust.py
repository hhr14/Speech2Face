import sys
weight_path = '/home/huirhuang/Speech2Face/data/weight.txt'
scale = [0]
mouth = [1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 15, 23, 24]
eye = [6, 7, 13, 14, 16, 17, 18, 19, 20, 21, 22]
pose = [25, 26, 27, 28, 29, 30, 31]
weight_scale = 1
weight_mouth = 1
weight_eye = 1
weight_pose = 1
f = open(weight_path, 'w')
for i in range(32):
    if i in scale:
        weight = weight_scale
    elif i in mouth:
        weight = weight_mouth
    elif i in eye:
        weight = weight_eye
    else:
        weight = weight_pose
    f.write(str(weight) + '\n')
