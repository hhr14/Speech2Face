if [ $# -ne 3 ] ; then
    echo "Usage input_fwh_npy_folder output_folder_path variance.npy"
    exit 1;
fi
temp_byte_path=/home/huirhuang/Speech2Face/tools/temp/temp_input.byte
temp_mlpg_byte_path=/home/huirhuang/Speech2Face/tools/temp/temp.mlpg_byte
temp_variance_path=/home/huirhuang/Speech2Face/tools/temp/temp.variance
temp_weighted_path=/home/huirhuang/Speech2Face/tools/temp/temp.weighted
cd lib
dir=$1
for file in $dir*; do
    python to_byte.py $file > $temp_byte_path
    python to_byte.py $3 > $temp_variance_path
    bash MLPG.sh $temp_byte_path $temp_mlpg_byte_path $temp_variance_path
    python to_text.py $2 $file 1 < $temp_mlpg_byte_path
    #if [ $4 == "1" ] ; then
    #    echo "use weight!"
    #    python add_weight.py $5 < $temp_mlpg_byte_path > $temp_weighted_path
    #    python to_text.py $2 $file < $temp_weighted_path
    #else
    #    python to_text.py $2 $file < $temp_mlpg_byte_path
    #fi
done
