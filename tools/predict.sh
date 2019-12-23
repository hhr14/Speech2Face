#!/usr/bin/env bash
if [[ $# != 8 && $# != 9 && $# != 11 ]] ; then
    echo -e "Usage:\n emotion [0: normal 1:angry 2:happy 3:sad]\n fwh_mode [MLPG, window]\n\
 from_name [0: False, 1:True]\n model_save_path [~/Speech2Face/weight/..]\n\
 dataset [~/Speech2Face/data/...]\n use_mfcc[0, 1, 2, t, s]\n \
[network_parameters]\n     --gpu\n     --network\n     --batch_size\n     \
--blstm_use_zoneout\n\n ------[Option]------\n is_16k\n window_length|sp_id\n\
 window_ne [0:False 1:True]\n add_after_window [0: no window >0: window length]"
    exit 1;
fi
output_ppg_folder=~/Speech2Face/sample/sample_ppg/
output_npy_folder=~/Speech2Face/predict_result/fwh_npy/
output_txt_folder=~/Speech2Face/predict_result/fwh_txt/
output_png_folder=~/Speech2Face/predict_result/fwh_png/
input_wave_folder=~/Speech2Face/sample/sample_wav/
output_mp4_folder=~/Speech2Face/predict_result/fwh_mp4/
variance=variance.npy
weight_path=~/Speech2Face/data/weight.txt
symbol=*
echo -e "\n>>>>>>>> start predict !\n"
bash wav2ppg_emotion.sh $input_wave_folder $output_ppg_folder $1 $3 $8 $6 $9
echo -e "\n>>>>>>>> wav2ppg_emotion finish !\n"
cd ~/Speech2Face/
python train.py --mode=predict --predict=sample/sample_ppg/ --output_folder=predict_result/fwh_npy/ --model_save_path=$4 --dataset=$5 ${7}
echo -e "\n>>>>>>>> predict fwh32 finish !\n"
cd ~/Speech2Face/tools/
if [ $2 == "MLPG" ] ; then
    bash fwh_MLPG.sh $output_npy_folder $output_txt_folder $5$variance
elif [[ $2 == "MLPG_var" ]] ; then
    bash fwh_MLPG_evar.sh $output_npy_folder $output_txt_folder $5
elif [ $2 == "window" ] ; then
    bash fwh_windowave_ne.sh $output_npy_folder $output_txt_folder $9 ${10} ${11}
else
    bash fwh_windowave_ne.sh $output_npy_folder $output_txt_folder 1 1 1
fi
echo -e "\n>>>>>>>> postprocess $2 finish !\n"
echo -e ">>>>>>>>> In current version we stop matlab part !\n"
rm -rf $output_ppg_folder$symbol
rm -rf $output_npy_folder$symbol
exit 1
for file in $output_txt_folder*; do
    cd ~/Speech2Face/tools/lib/render_txt-gl-fwh32b/
    bash ~/MATLAB/bin/matlab -r "run $file $output_png_folder"
    cd ~/Speech2Face/tools/
    bash generate_animation.sh $output_png_folder $input_wave_folder $output_mp4_folder 100
    echo -e "\n>>>>>>>> $file animation finish !\n"
    rm -rf $output_png_folder$symbol
done
echo -e "\n>>>>>>>> predict finish !\n"
#rm -rf $output_txt_folder$symbol

