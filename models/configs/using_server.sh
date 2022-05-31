
 screen # This lets you run the command in the background and close the console window.
 cd ~/models/optimize
 python3 hyperbones.py > optimize_log.txt 2>&1 &

 screen
# run whatever commands in background
# press ctrl-A, ctrl-D
# that will leave screen mode
# you can close the window
# later on, open ssh again
 screen -r # resumes