# !pip install youtube-search-python
# !pip install yt-dlp

import os
import json
import yt_dlp
from youtubesearchpython import VideosSearch

def load_downloaded_videos(tracking_file: str):
    """Load list of already downloaded video IDs."""
    if os.path.exists(tracking_file):
        with open(tracking_file, 'r') as f:
            return json.load(f)
    return []

def save_downloaded_videos(tracking_file: str, video_ids: list):
    """Save list of downloaded video IDs."""
    with open(tracking_file, 'w') as f:
        json.dump(video_ids, f, indent=2)

def search_and_download_videos(query: str, limit: int, output_dir: str = 'videos', tracking_file: str = None):
    """
    Searches for videos on YouTube and downloads them automatically, avoiding duplicates.

    Args:
        query (str): The search term (e.g., "cat scratching furniture").
        limit (int): The maximum number of videos to download.
        output_dir (str): The directory to save the downloaded videos.
        tracking_file (str): JSON file to track downloaded video IDs.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Load previously downloaded video IDs
    downloaded_ids = []
    if tracking_file:
        downloaded_ids = load_downloaded_videos(tracking_file)
        print(f"Found {len(downloaded_ids)} previously downloaded videos")

    print(f"🔍 Searching for '{query}' videos...")
    # Perform the search and limit the results
    videos_search = VideosSearch(query, limit=limit)
    search_results = videos_search.result()['result']

    if not search_results:
        print("No videos found for your query.")
        return

    print(f"Found {len(search_results)} videos. Starting download...")

    # Configure download options for yt-dlp
    output_template = os.path.join(output_dir, '%(title)s [%(id)s].%(ext)s')
    ydl_opts = {
        'format': 'best[height>=480][ext=mp4]/best[ext=mp4]',  # High quality preference
        'outtmpl': output_template,
        'quiet': True,
        'merge_output_format': 'mp4',
        'writesubtitles': False,
        'writeautomaticsub': False,
    }

    new_downloads = []
    downloaded_count = 0
    
    for video in search_results:
        video_url = video['link']
        video_title = video['title']
        video_id = video['id']
        
        # Skip if already downloaded
        if video_id in downloaded_ids:
            print(f"⏭️  Skipping '{video_title}' - already downloaded")
            continue
            
        print(f"\nDownloading '{video_title}'...")
        print(f"URL: {video_url}")

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            print(f"✅ Successfully downloaded '{video_title}'")
            new_downloads.append(video_id)
            downloaded_count += 1
        except yt_dlp.utils.DownloadError as e:
            print(f"❌ Failed to download '{video_title}'. Reason: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    
    # Update tracking file
    if tracking_file and new_downloads:
        updated_ids = downloaded_ids + new_downloads
        save_downloaded_videos(tracking_file, updated_ids)
        print(f"\n📝 Updated tracking file with {len(new_downloads)} new downloads")
    
    print(f"\n📊 Downloaded {downloaded_count} new videos out of {len(search_results)} found")

if __name__ == '__main__':
    search_query = "real funny cat playing"
    number_of_videos_to_download = 14
    download_folder = "./cat_action_videos/playing_raw"
    tracking_file = "./cat_action_videos/downloaded_videos.json"
    
    os.makedirs(download_folder, exist_ok=True)
    os.makedirs(os.path.dirname(tracking_file), exist_ok=True)

    search_and_download_videos(
        query=search_query, 
        limit=number_of_videos_to_download,
        output_dir=download_folder,
        tracking_file=tracking_file
    )
    print("\nAll tasks complete.")