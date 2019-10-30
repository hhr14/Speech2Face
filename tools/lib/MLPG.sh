mlpg_path=/home/huirhuang/upload/SPTK-3.11/bin/mlpg
delta_path=/home/huirhuang/upload/SPTK-3.11/bin/delta
merge_path=/home/huirhuang/upload/SPTK-3.11/bin/merge
vstat_path=/home/huirhuang/upload/SPTK-3.11/bin/vstat
temp_covariance_path=/home/huirhuang/Speech2Face/tools/temp/temp.covariance
temp_cov_path=/home/huirhuang/Speech2Face/tools/temp/temp.cov
temp_pdf_path=/home/huirhuang/Speech2Face/tools/temp/temp.pdf
temp_mlpg_path=/home/huirhuang/Speech2Face/tools/temp/temp.mlpg
#cat $1 | \
#    vstat -d -l 96 -o 2 | \
#    cat > $temp_covariance_path
num_frames=$(( $(stat -c %s $1) / 384 ))
for n in $(seq $num_frames); do \
#    cat $temp_covariance_path
    cat $3
done | \
    cat >$temp_cov_path
cat $temp_cov_path | \
    merge -l 96 -L 96 $1 | \
    cat >$temp_pdf_path
cat $temp_pdf_path | \
    mlpg -m 31 -d -0.5 0 0.5 -d 1.0 -2.0 1.0 | \
    cat >$2
