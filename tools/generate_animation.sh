if [ $# != 4 ] ; then
    echo "Usage input_png input_wav_folder mp4_output_folder sample_rate(fps)"
    exit 1;
fi
#dir=$1
#for png_folder in $dir*; do
python lib/get_file_prefix.py $1 | bash lib/ffmpeg_generate.sh $2 $3 $4
#done
