
{{ config(materialized='table') }}

with source_videos as (
    select * from {{ source('Youtube_LLM_Data', 'videos') }}

),

final as (

    SELECT
  CHANNEL_ID,
  VIDEO_ID,
  CHANNELTITLE,
  TITLE,
  TAGS,
  PUBLISHEDAT,
  VIEWCOUNT,
  DEFAULTAUDIOLANGUAGE,
  COMMENTCOUNT,
  LIKECOUNT,
  DURATION,
  DEFINITION,
  CAPTION
    FROM  source_videos
)

select * from final
