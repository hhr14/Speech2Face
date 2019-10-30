delta_path=/home/huirhuang/upload/SPTK-3.11/bin/delta
temp_delta_path=/home/huirhuang/Speech2Face/tools/temp/temp.delta
cd $delta_path
cat $1 | \
    ./delta -m 31 -d -0.5 0 0.5 -d 1 -2 1 | \
    cat > $temp_delta_path
