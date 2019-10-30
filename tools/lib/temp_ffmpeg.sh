ffmpeg -nostdin -r 60 -i ~/test_sing_140_png/test.fwh32b_%06d.png \
    -i ~/input_wav/sample.wav \
    -vf "drawtext=fontfile=/home/huirhuang/upload/chinese.msyh.ttf:fontsize=40:fontcolor=white:x=(w-text_w)/2:y=(h-text_h-10):box=1:boxcolor=black@0.6:boxborderw=10:enable='between(t,0,0.5)':text='"sample_140.mp4"'" \
    -map 0:v:0 -map 1:a:0 -pix_fmt yuv420p -y ~/input_wav/sample_140.mp4
