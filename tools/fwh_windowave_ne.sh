if [ $# != 5 ] ; then
    echo "Usage: bash *.sh input_fwh_npy_folder output_folder window_length window_ne[0: jump, 1: ne] add_after_window"
    exit 1;
fi
temp_winave_byte_path=/home/huirhuang/Speech2Face/tools/temp/temp_winave.byte
temp_weighted_path=/home/huirhuang/Speech2Face/tools/temp/temp.weighted
cd lib
dir=$1
for file in $dir*; do
    python fwh_window_ne.py $file $3 $4 $5 > $temp_winave_byte_path
    python to_text.py $2 $file 1 < $temp_winave_byte_path
    #if [ $5 == "1" ] ; then
    #    echo "use_weight!"
    #    python add_weight.py $6 < $temp_winave_byte_path > $temp_weighted_path
    #    python to_text.py $2 $file < $temp_weighted_path
    #else
    #    python to_text.py $2 $file < $temp_winave_byte_path
    #fi
done
