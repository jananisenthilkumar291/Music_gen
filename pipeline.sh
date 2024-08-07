python3 a_fetch_trending_playlists.py
mkdir tracks
python3 download_tracks.py
mkdir wav_tracks

python3 convert_mp3_to_wav.py
mv tracks/*.wav wav_tracks/
python3 temp_process_wav.py
python3 building_patterns1.py
wget https://www.dropbox.com/s/4x27l49kxcwamp5/GeneralUser_GS_1.471.zip
unzip GeneralUser_GS_1.471.zip
fluidsynth -ni './GeneralUser GS 1.471/GeneralUser GS v1.471.sf2' generated_music.mid -F generated_music.wav -r 44100
ffmpeg -i generated_music.wav generated_music.mp3
