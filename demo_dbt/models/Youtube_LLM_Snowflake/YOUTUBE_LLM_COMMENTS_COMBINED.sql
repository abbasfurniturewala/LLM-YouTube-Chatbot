{{ config(materialized='table') }}

with source_comments_combined as (
        select * from {{ source('Youtube_LLM_Data', 'comments_combined') }}

),

final as ( 
    SELECT * FROM  source_comments_combined
    )

select * from final