if [ $# != 5 ] ; then
    echo "Usage input_wav output_folder emotion[0:normal 1:angry 2:happy 3:sad] split_time[s] overlap[s]"
    echo "Only support for single wav_file"
    exit 1;
fi
temp_ppg_folder=~/Speech2Face/tools/temp/split_ppg/
temp_wav_folder=~/Speech2Face/tools/temp/split_wav/
temp_itp_ppg_folder=~/Speech2Face/tools/temp/itp_ppg/
temp_ppg=~/Speech2Face/tools/temp/temp.ppg
mkdir $temp_wav_folder
mkdir $temp_ppg_folder
mkdir $temp_itp_ppg_folder
cd lib/
python split_wav_scipyio.py $1 $temp_wav_folder $4 $5
#file_kind='.ppg'
temp_wav_file_list=`ls $temp_wav_folder`
for wav in $temp_wav_file_list
do
    echo $wav
    #ppg_name=${wav%.wav}$file_kind
    #echo $ppg_name
    ./wav2ppg data/ppg.model data/fbank.cfg data/ed.cfg $temp_wav_folder$wav $temp_ppg
    python byte_to_npy.py $temp_wav_folder$wav $temp_ppg_folder < $temp_ppg
done
python ppg_postprocess.py $temp_ppg_folder $temp_itp_ppg_folder $5 100
for temp_ppg in $temp_itp_ppg_folder*; do
    python add_emotion_from_input.py $3 $temp_ppg $2 0
done
rm -rf $temp_wav_folder
rm -rf $temp_ppg_folder
rm -rf $temp_itp_ppg_folder
