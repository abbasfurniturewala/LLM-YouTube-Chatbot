import pandas as pd
import numpy as np
from dateutil import parser
import isodate
from datetime import datetime, timedelta
from googleapiclient.errors import HttpError 

# Data visualization libraries
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
sns.set(style="darkgrid", color_codes=True)

# Google API
from googleapiclient.discovery import build

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
from wordcloud import WordCloud

import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
import pandas as pd

channel_ids = [
    'UCupvZG-5ko_eiXAupbDfxWw',  # CNN
     'UCXIJgqnII2ZOINSWNOGFThA',  # FOX NEWS
     'UCaXkIU1QidjPwiAYu6GcHjg',  # MSNBC
     'UCBi2mrWuNuyYy4gbM6fU18Q',  # ABC NEWS
     'UC8p1vwvWtl6T73JiExfWs1g',  # CBS NEWS
]
api_key = 'AIzaSyB-4NIQtecQPbRX7TWKphThkb9_Brh2wL4' 
youtube = build('youtube', 'v3', developerKey=api_key)

def get_channel_stats(youtube, channel_ids):
    """
    Get channel statistics: title, subscriber count, view count, video count, upload playlist
    Params:
    
    youtube: the build object from googleapiclient.discovery
    channels_ids: list of channel IDs
    
    Returns:
    Dataframe containing the channel statistics for all channels in the provided list: title, subscriber count, view count, video count, upload playlist
    
    """
    all_data = []
    request = youtube.channels().list(
                part='snippet,contentDetails,statistics',
                id=','.join(channel_ids))
    response = request.execute() 
    
    for i in range(len(response['items'])):
        data = dict(channelName = response['items'][i]['snippet']['title'],
                    channel_id=channel_ids[i],
                    subscribers = response['items'][i]['statistics']['subscriberCount'],
                    views = response['items'][i]['statistics']['viewCount'],
                    totalVideos = response['items'][i]['statistics']['videoCount'],
                    playlistId = response['items'][i]['contentDetails']['relatedPlaylists']['uploads'])
        all_data.append(data)
    
    return pd.DataFrame(all_data)

def get_video_ids(youtube, playlist_id):
    """
    Get list of video IDs of all videos in the given playlist for the last month past(30 days) 
    Params:
    
    youtube: the build object from googleapiclient.discovery
    playlist_id: playlist ID of the channel
    
    Returns:
    List of video IDs of all videos in the playlist
    
    """
    one_month_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%dT%H:%M:%SZ')
    
    request = youtube.playlistItems().list(
                part='contentDetails',
                playlistId=playlist_id,
                maxResults=50)
    response = request.execute()
    
    video_ids = []
    
    for i in range(len(response['items'])):
        video_published_at = response['items'][i]['contentDetails']['videoPublishedAt']
        
        # Check if the video was published in the past month
        if video_published_at >= one_month_ago:
            video_ids.append(response['items'][i]['contentDetails']['videoId'])
    

    next_page_token = response.get('nextPageToken')
    more_pages = True
    
    while more_pages:
        if next_page_token is None:
            more_pages = False
        else:
            request = youtube.playlistItems().list(
                        part='contentDetails',
                        playlistId = playlist_id,
                        maxResults = 50,
                        pageToken = next_page_token
                        )
            response = request.execute()
    
            for i in range(len(response['items'])):
                video_published_at = response['items'][i]['contentDetails']['videoPublishedAt']
        
                # Check if the video was published in the past month
                if video_published_at >= one_month_ago:
                    video_ids.append(response['items'][i]['contentDetails']['videoId'])
            
            next_page_token = response.get('nextPageToken')
    return video_ids



def get_video_details(youtube, video_ids, channel_id):
    """
    Get video statistics of all videos with given IDs
    Params:
    
    youtube: the build object from googleapiclient.discovery
    video_ids: list of video IDs
    channel_id: ID of the channel
    
    Returns:
    Dataframe with statistics of videos, i.e.:
        'channel_id', 'channelTitle', 'title', 'description', 'tags', 'publishedAt'
        'viewCount', 'likeCount', 'favoriteCount', 'commentCount'
        'duration', 'definition', 'caption'
    """
        
    all_video_info = []
    
    for i in range(0, len(video_ids), 50):
        request = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=','.join(video_ids[i:i+50])
        )
        response = request.execute() 

        for video in response['items']:
            stats_to_keep = {'snippet': ['channelTitle', 'title', 'description', 'tags', 'publishedAt','defaultAudioLanguage'],
                             'statistics': ['viewCount', 'likeCount', 'favouriteCount', 'commentCount'],
                             'contentDetails': ['duration', 'definition', 'caption']
                            }
            video_info = {}
            video_info['channel_id'] = channel_id  # Add channel_id to the DataFrame
            video_info['video_id'] = video['id']

            for k in stats_to_keep.keys():
                for v in stats_to_keep[k]:
                    try:
                        video_info[v] = video[k][v]
                    except:
                        video_info[v] = None

            all_video_info.append(video_info)
            
    return pd.DataFrame(all_video_info)



def get_playlists_info(youtube, channel_ids):


    all_playlist_data = []

    """
    Retreiving Playlist data for all the channels
    
    """
    
    for channel_id in channel_ids:
        request = youtube.playlists().list(
            part="snippet",
            channelId=channel_id,
            maxResults=50  # Adjust the maximum number of playlists to retrieve if needed
        )
        response = request.execute()

        for playlist in response.get("items", []):
            playlist_data = dict(
                playlist_id=playlist["id"],
                title=playlist["snippet"]["title"],
                description=playlist["snippet"]["description"],
                publishedAt=playlist["snippet"]["publishedAt"],
                channelId=playlist["snippet"]["channelId"],
                channelTitle=playlist["snippet"]["channelTitle"],
                defaultLanguage=playlist["snippet"].get("defaultLanguage"),
                thumbnailUrl=playlist["snippet"]["thumbnails"]["default"]["url"]
            )
            all_playlist_data.append(playlist_data)
    return pd.DataFrame(all_playlist_data)


def get_captions(youtube, video_ids):
    caption_list = []

    for video_i in video_ids:
            captions = youtube.captions().list(
            part="snippet",
            videoId=video_i
        ).execute()

        # List to store comments as dictionaries
            

        # Extract comments and append them to the list
            for caption in captions["items"]:
                snippet = caption["snippet"]
                caption_dict = {
        "videoId": snippet["videoId"],
        "lastUpdated": snippet["lastUpdated"],
        "trackKind": snippet["trackKind"],
        "language": snippet["language"],
        "name": snippet["name"],
        "audioTrackType": snippet["audioTrackType"],
        "status": snippet["status"]
    }
                caption_list.append(caption_dict)
    return(pd.DataFrame(caption_list))

def get_comments(youtube, video_ids):
    """
    Get top level comments as text from all videos with given IDs (only the first 50 comments per video due to quote limit of Youtube API)
    Params:
    
    youtube: the build object from googleapiclient.discovery
    video_ids: list of video IDs
    
    Returns:
    Dataframe with video IDs and associated top level comment in text.
    
    """
    all_comments = []
    all_comments_data = []    
    for video_id in video_ids:
        comments_in_video_info = {}
        try:   
            request = youtube.commentThreads().list(
                part="snippet,replies",
                videoId=video_id
            )
            response = request.execute()
            
            comments_in_video= []
            comments_in_video_info = {}
            for comment in response['items'][:50]:
                comment_text = comment['snippet']['topLevelComment']['snippet']['textOriginal']
        
                # Append the comment text to the list
                comments_in_video.append(comment_text)
                comments_data = {'video_id': video_id, 
                                'comments': comment_text,
                                'likeCount': comment['snippet']['topLevelComment']['snippet']['likeCount'],
                                'authorDisplayName': comment['snippet']['topLevelComment']['snippet']['authorDisplayName'],
                                'authorProfileImageUrl': comment['snippet']['topLevelComment']['snippet']['authorProfileImageUrl'],
                                'authorChannelUrl': comment['snippet']['topLevelComment']['snippet']['authorChannelUrl'],
                                'authorChannelId': comment['snippet']['topLevelComment']['snippet']['authorChannelId']['value'],
                                'channelId': comment['snippet']['topLevelComment']['snippet']['channelId'],
                                'canRate': comment['snippet']['topLevelComment']['snippet']['canRate'],
                                'viewerRating': comment['snippet']['topLevelComment']['snippet']['viewerRating'],
                                'publishedAt': comment['snippet']['topLevelComment']['snippet']['publishedAt']
                                
                                                            
                                }
                all_comments_data.append(comments_data)
            comments_in_video_info = {'video_id': video_id, 'comments': comments_in_video}
            


        except: 
            # When error occurs - most likely because comments are disabled on a video
            print('Could not get comments for video ' + video_id)



        all_comments.append(comments_in_video_info)

            
                
        
        # Create a dictionary for each comment and append it to the list
                # comment_info = {'video_id': video_id, 'comment': comment_text}
                # comments_in_video_info.append(comment_info)
        

        
        
    
        
    return pd.DataFrame(all_comments_data) , pd.DataFrame(all_comments)   


def fetch_data(api_key, video_ids, channel_id):
    youtube = build('youtube', 'v3', developerKey=api_key)
    

    # Get video data for the current chunk of video IDs
    video_data = get_video_details(youtube, video_ids, channel_id)

    # Get comment data for the current chunk of video IDs
    comments_data_df, comments_combined_df = get_comments(youtube, video_ids)

    return video_data, comments_data_df, comments_combined_df



api_keys = ['AIzaSyA4Sd1FkOSah19dL7cg7OuBUj9VBJiE2fE', 'AIzaSyB-4NIQtecQPbRX7TWKphThkb9_Brh2wL4']

channel_df = get_channel_stats(youtube, channel_ids)

snowflake_user = 'FURNITUREWALAABBAS'
snowflake_password = 'Abba$123'
snowflake_account = 'jrnvcvi-sw72415'
snowflake_database = 'YOUTUBE_LLM'
snowflake_schema = 'RAW'
#snowflake_warehouse = 'your_warehouse'

# Create a Snowflake connection
conn = snowflake.connector.connect(
    user=snowflake_user,
    password=snowflake_password,
    account=snowflake_account,
    #warehouse=snowflake_warehouse,
    database=snowflake_database,
    schema=snowflake_schema
)

# Create a cursor object
cur = conn.cursor()

# CREATING TABLE CHANNELS

cur.execute("create or replace TABLE YOUTUBE_LLM.RAW.CHANNELS ( \
	CHANNELNAME VARCHAR(16777216),\
	CHANNEL_ID VARCHAR(16777216),\
	SUBSCRIBERS NUMBER(38,0),\
	VIEWS NUMBER(38,0),\
	TOTALVIDEOS NUMBER(38,0),\
	PLAYLISTID VARCHAR(16777216) )")


write_pandas(conn, channel_df, 'CHANNELS', quote_identifiers= False)


conn.commit()

# Close the cursor and connection
cur.close()
conn.close()
