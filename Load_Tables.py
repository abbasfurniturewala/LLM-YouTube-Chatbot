import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
import pandas as pd

# Snowflake connection parameters
snowflake_user = 'FURNITUREWALAABBAS'
snowflake_password = 'Abba$123'
snowflake_account = 'jrnvcvi-sw72415'
snowflake_database = 'YOUTUBE_LLM'
snowflake_schema = 'PUBLIC'
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

# CREATING TABLE VIDEOS

cur.execute("CREATE OR REPLACE TABLE YOUTUBE_LLM.PUBLIC.VIDEOS ( \
    channel_id STRING, \
    video_id STRING,\
    channelTitle STRING,\
    title STRING,\
    description STRING,\
    publishedAt TIMESTAMP_NTZ,\
    viewCount FLOAT,\
    likeCount FLOAT,\
    favouriteCount FLOAT,\
    commentCount FLOAT,\
    duration STRING, \
    definition STRING,\
    caption BOOLEAN,\
    pushblishDayName STRING,\
    durationSecs FLOAT,\
    tagsCount INTEGER,\
    likeRatio FLOAT,\
    commentRatio FLOAT,\
    titleLength INTEGER)" ) 

# LOAD TABLE VIDEOS
 
write_pandas(conn, videos_df, 'VIDEOS', quote_identifiers= False)

# CREATING TABLE COMMENTS

cur.execute("CREATE OR REPLACE TABLE YOUTUBE_LLM.PUBLIC.COMMENTS ( \
    VIDEO_ID VARCHAR(1000000) ,  \
    COMMENTS VARCHAR(16777216),  \
    LIKECOUNT NUMBER(38, 0), \
    AUTHORDISPLAYNAME VARCHAR(500), \
    AUTHORPROFILEIMAGEURL VARCHAR(10000),\
    AUTHORCHANNELURL VARCHAR(10000),\
    AUTHORCHANNELID VARCHAR(10000),\
    CHANNELID VARCHAR(10000),\
    CANRATE BOOLEAN,\
    VIEWERRATING VARCHAR(10000),\
    PUBLISHEDAT TIMESTAMP_NTZ(9))")


write_pandas(conn, comments_all_data_df, 'COMMENTS', quote_identifiers= False)

# CREATING TABLE COMMENTS_COMBINED

cur.execute("create or replace TABLE YOUTUBE_LLM.PUBLIC.COMMENTS_COMBINED ( \
	VIDEO_ID VARCHAR(10000000),\
	COMMENTS VARIANT);")

write_pandas(conn, comments__df, 'COMMENTS_COMBINED', quote_identifiers= False)


# CREATING TABLE CHANNELS

cur.execute("create or replace TABLE YOUTUBE_LLM.PUBLIC.CHANNELS ( \
	CHANNELNAME VARCHAR(16777216),\
	CHANNEL_ID VARCHAR(16777216),\
	SUBSCRIBERS NUMBER(38,0),\
	VIEWS NUMBER(38,0),\
	TOTALVIDEOS NUMBER(38,0),\
	PLAYLISTID VARCHAR(16777216) )")


write_pandas(conn, channel_df, 'CHANNELS', quote_identifiers= False)

# CREATING TABLE PLAYLIST

cur.execute("create or replace TABLE YOUTUBE_LLM.PUBLIC.PLAYLIST (\
	PLAYLIST_ID VARCHAR(10000000),\
	TITLE VARCHAR(10000000),\
	DESCRIPTION VARCHAR(10000000),\
	PUBLISHEDAT TIMESTAMP_NTZ(9),\
	CHANNELID VARCHAR(10000000),\
	CHANNELTITLE VARCHAR(10000000),\
	DEFAULTLANGUAGE VARCHAR(10000000),\
	THUMBNAILURL VARCHAR(10000000));")


write_pandas(conn, playlist_df, 'PLAYLIST', quote_identifiers= False)

# Commit the changes
conn.commit()

# Close the cursor and connection
cur.close()
conn.close()