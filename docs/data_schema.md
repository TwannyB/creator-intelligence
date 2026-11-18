# Data Schema

## Video data

| Field | Description |
|---|---|
| id | TikTok video ID |
| create_time | Video publication timestamp |
| video_description | Caption and hashtags |
| duration | Video length in seconds |
| view_count | Total views |
| like_count | Total likes |
| comment_count | Total comments |
| share_count | Total shares |

## Aggregated commerce data

| Field | Description |
|---|---|
| Content ID | TikTok video identifier |
| unique_orders | Number of unique orders attributed to the video |
| items_sold | Total attributed items sold |
| items_refunded | Total refunded items |
| total_gmv | Total attributed gross merchandise value |
| unique_products | Number of products sold |
| standard_commission | Total standard commission |
| shop_ads_commission | Total Shop Ads commission |

## Derived labels

| Field | Description |
|---|---|
| is_shop_video | Whether the TikTok promotes a Shop/affiliate product |
| made_sale | Whether the video generated at least one order |
| orders_per_1000_views | Orders normalized by views |
| gmv_per_1000_views | GMV normalized by views |