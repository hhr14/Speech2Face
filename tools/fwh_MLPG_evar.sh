#!/usr/bin/env bash
if [[ $# -ne 3 ]] ; then
    echo "Usage input_fwh_npy_folder output_folder_path dataset_folder"
    exit 1;
fi
temp_byte_path=/home/huirhuang/Speech2Face/tools/temp/temp_input.byte
temp_mlpg_byte_path=/home/huirhuang/Speech2Face/tools/temp/temp.mlpg_byte
temp_weighted_path=/home/huirhuang/Speech2Face/tools/temp/temp.weighted
temp_emotion_var=~/Speech2Face/tools/temp/temp_emotion_var.npy
temp_variance_path=/home/huirhuang/Speech2Face/tools/temp/temp.variance
cd lib
dir=$1
for file in $dir*; do
    python to_byte.py $file > $temp_byte_path
    python select_var.py $3 $temp_emotion_var $file
    python to_byte.py $temp_emotion_var > $temp_variance_path
    bash MLPG.sh $temp_byte_path $temp_mlpg_byte_path $temp_variance_path
    python to_text.py $2 $file 1 < $temp_mlpg_byte_path
done
rm $temp_emotion_var
rm $temp_byte_path
rm $temp_mlpg_byte_path
rm $temp_variance_path
