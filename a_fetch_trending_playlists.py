import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import json

# Replace with your own Spotify Developer credentials
client_id = '49cf660e607d4faeafbf8781e467e89b'
client_secret = 'ef16dc7e7e6343c89818a8518088cfae'

# Authenticate with Spotify
credentials = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=credentials)

# Function to get top playlists
def get_top_playlists(limit=5):
    results = sp.featured_playlists(limit=limit)
    playlists = results['playlists']['items']
    return playlists

# Function to get tracks from a playlist
def get_playlist_tracks(playlist_id):
    results = sp.playlist_tracks(playlist_id)
    tracks = results['items']
    return tracks

# Get top 5 trending playlists
playlists = get_top_playlists()

# Print playlist names and get tracks
count_of_tracks = 0
for playlist in playlists:
    print(f"Playlist: {playlist['name']}")
    playlist_id = playlist['id']
    tracks = get_playlist_tracks(playlist_id)
    count_of_tracks += len(tracks)
    for track in tracks:
        track_name = track['track']['name']
        artists = [artist['name'] for artist in track['track']['artists']]
        print(f"Track: {track_name} by {', '.join(artists)}")
    print("\n")

print(f'Downloading {count_of_tracks} tracks from top playlists\n')

# Save playlist data to a file
with open('top_playlists.json', 'w') as f:
    json.dump(playlists, f, indent=4)

