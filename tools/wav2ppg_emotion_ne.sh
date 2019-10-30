if [[ $# != 5 && $# != 6 ]] ; then
    echo -e "Usage input_wav_folder output_folder emotion[0:normal 1:angry 2:happy 3:sad] from_name[0:no 1:yes] is_16k[0:False 1:True]\n\n[Option]\nuse_mfcc[0, 1]"
    exit 1;
fi
temp_ppg_path=/home/huirhuang/Speech2Face/tools/temp/temp.ppg
temp_16k_path=/home/huirhuang/Speech2Face/tools/temp/temp16k.wav
temp_mfcc_path=/home/huirhuang/Speech2Face/tools/temp/temp.mfcc
temp_frame_path=/home/huirhuang/Speech2Face/tools/temp/temp.frame
temp_folder=/home/huirhuang/Speech2Face/tools/temp/
temp_mfcc_npy=/home/huirhuang/Speech2Face/tools/temp/temp.npy
temp_wav_byte=/home/huirhuang/Speech2Face/tools/temp/temp_wav.byte
temp_ppg_energy=/home/huirhuang/Speech2Face/tools/temp/temp.energy
cd lib/
dir=$1
for file in $dir*; do
    if [ $5 == "0" ] ; then
        ffmpeg -i $file -ar 16000 -ac 1 $temp_16k_path
        python wav2byte.py $temp_16k_path > $temp_wav_byte
        rm $temp_16k_path
    else
        python wav2byte.py $file > $temp_wav_byte
    fi
    frame -l 640 -p 160 $temp_wav_byte > $temp_frame_path
    #mfcc -l 640 -m 12 -s 16 -E > $temp_mfcc_path < $temp_frame_path
    if [ $6 == "1" ] ; then
        mfcc -l 640 -m 40 -s 16 -E > $temp_mfcc_path < $temp_frame_path
        python add_emotion_from_input.py $3 $file $2 1 $4 41 < $temp_mfcc_path
    else
        mfcc -l 640 -m 12 -s 16 -E > $temp_mfcc_path < $temp_frame_path
        bash wav2ppg_pro.sh $file $temp_ppg_path
        #python byte_to_npy.py $temp_mfcc_path $temp_folder < $temp_mfcc_path
        #python add_energy.py $temp_mfcc_npy < $temp_ppg_path > $temp_ppg_energy
        python add_emotion_from_input.py $3 $file $2 1 $4 218 < $temp_ppg_path
    fi
done
rm $temp_ppg_path
rm $temp_mfcc_path
rm $temp_frame_path
rm $temp_mfcc_npy
rm $temp_wav_byte
