{{ config(materialized='table') }}

with source_channels as (
        select * from {{ source('Youtube_LLM_Data', 'channels') }}

),

final as ( 
    SELECT * FROM  source_channels
    )

select * from final