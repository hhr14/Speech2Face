read png_file_path mp4_name wave_name
ffmpeg_static_path=/home/huirhuang/upload/ffmpeg-git-20190809-amd64-static/
cd $ffmpeg_static_path
./ffmpeg -nostdin -r $3 -i $png_file_path \
    -i $1$wave_name \
    -vf "drawtext=fontfile=/home/huirhuang/upload/chinese.msyh.ttf:fontsize=40:fontcolor=white:x=(w-text_w)/2:y=(h-text_h-10):box=1:boxcolor=black@0.6:boxborderw=10:enable='between(t,0,0.5)':text='"$mp4_name"'" \
    -map 0:v:0 -map 1:a:0 -pix_fmt yuv420p -y $2$mp4_name
