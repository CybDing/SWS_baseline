echo "[*] Starting the remote control..."

echo "[*] Starting the cat regonition program"
osascript -e 'tell app "Terminal" to do script "source /opt/anaconda3/etc/profile.d/conda.sh && conda activate sws3009 && cd /Users/ding/Desktop/NUS-proj/advanced/CatTracker && python3 ./video-processing-threaded.py"'

echo "[*] Starting the cat behavior analysis program"
osascript -e 'tell app "Terminal" to do script "source /opt/anaconda3/etc/profile.d/conda.sh && conda activate sws3009 && cd /Users/ding/Desktop/NUS-proj/advanced/Action_Detector && python ./behavior_analysis.py"'

sleep 5
cd /Users/ding/Desktop/NUS-proj/
echo "[*] Starting the remote control dashboard"
open ./remote_control_update_new.html 

echo "[^] All programmes are being set up"