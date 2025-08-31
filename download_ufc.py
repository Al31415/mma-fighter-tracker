import requests
from bs4 import BeautifulSoup
import datetime
from dateutil import parser as dateparser
import os
import subprocess
import re

# Set up
base_url = "https://watchmmafull.com"
search_base = f"{base_url}/search/ufc/"
output_dir = "ufc_videos"
os.makedirs(output_dir, exist_ok=True)

current_date = datetime.datetime.now()
start_date = current_date - datetime.timedelta(days=365 * 10)

def safe_filename(title):
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', title) + ".mp4"

# Find total pages (from previous knowledge, 142)
total_pages = 142

for page in range(1, total_pages + 1):
    page_url = f"{search_base}{page}"
    response = requests.get(page_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    for item in soup.find_all('li', class_='item-movie'):
        a_tag = item.find('a')
        fight_url = base_url + a_tag['href']
        title = a_tag['title']
        
        # Parse date from title
        try:
            date_str = title.split(' - ')[-1]
            fight_date = dateparser.parse(date_str)
            if start_date <= fight_date <= current_date:
                # Get fight page
                fight_response = requests.get(fight_url)
                fight_soup = BeautifulSoup(fight_response.text, 'html.parser')
                
                # Find iframe src
                iframe = fight_soup.find('iframe')
                if iframe:
                    video_src = iframe['src']
                    
                    # Download using yt-dlp
                    output_path = os.path.join(output_dir, safe_filename(title))
                    cmd = ['yt-dlp', video_src, '-o', output_path]
                    subprocess.run(cmd)
                    print(f"Downloaded: {title}")
        except Exception as e:
            print(f"Error processing {title}: {e}") 