{{ config(materialized='table') }}

with source_playlist as (
        select * from {{ source('Youtube_LLM_Data', 'playlist') }}

),

final as ( 
SELECT  
PLAYLIST_ID,
TITLE,
DESCRIPTION,
PUBLISHEDAT,
CHANNELID,
CHANNELTITLE,
THUMBNAILURL
FROM  source_playlist
    )

select * from final