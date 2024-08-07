Step1:
a_fetch_trending_playlists.py
	- use the file to retrieve top 5 trending playlists on spotify
	- you need to have spotify for developers account to do this.
	- replace cliet_id and client_secret wherever needed.
	- the script creates a json file with playlist data called top_playlists.json
mkdir tracks
Step2:
download_tracks.py
	- Use the file to download tracks from the playlists from the top_playlists.json file.
	- These mp3 s are saved in the tracks folder.

mkdir wav_tracks
Step3:
convert_mp3_to_wav.py
	- Use this file to convert mp3 to wav files.
	- these wav tracks are saved in wav_tracks folder
tep 4:
Get numerical data from .wav file.
	- Extraction of Chroma Features.
	- temp_process_wav.py
drwxrwxrwx 1 janani0313 janani0313 4096 Jun 27 21:11 might_not_need
-rwxrwxrwx 1 janani0313 janani0313 1088 Jun 23 13:36 temp_process_wav.py
To convert .mid produced by builing_patterns.py
- wget https://www.dropbox.com/s/4x27l49kxcwamp5/GeneralUser_GS_1.471.zip
- unzip GeneralUser_GS_1.471.zip
-  fluidsynth -ni './GeneralUser GS 1.471/GeneralUser GS v1.471.sf2' generated_music.mid -F generated_music.wav -r 44100
-  ffmpeg -i generated_music.wav generated_music.mp3


python3 a_fetch_trending_playlists.py
mkdir tracks
python3 download_tracks.py
mkdir wav_tracks
python3 convert_mp3_to_wav.py
python3 temp_process_wav.py
python3 building_patterns1.py
wget https://www.dropbox.com/s/4x27l49kxcwamp5/GeneralUser_GS_1.471.zip
unzip GeneralUser_GS_1.471.zip
fluidsynth -ni './GeneralUser GS 1.471/GeneralUser GS v1.471.sf2' generated_music.mid -F generated_music.wav -r 44100
ffmpeg -i generated_music.wav generated_music.mp3


