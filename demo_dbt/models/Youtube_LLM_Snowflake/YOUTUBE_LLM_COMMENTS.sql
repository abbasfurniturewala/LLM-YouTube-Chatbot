{{ config(materialized='table') }}

with source_comments as (
        select * from {{ source('Youtube_LLM_Data', 'comments') }}

),

final as ( 
    SELECT 
    VIDEO_ID,
COMMENTS,
LIKECOUNT,
AUTHORDISPLAYNAME,
AUTHORPROFILEIMAGEURL,
AUTHORCHANNELURL,
AUTHORCHANNELID,
CHANNELID,
CANRATE,
PUBLISHEDAT 
    FROM  source_comments
    )

select * from final
        