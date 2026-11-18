# Raw Data

This folder contains source data used by the Creator Intelligence pipeline.

The actual datasets are excluded from GitHub because they contain private creator and commerce information.

Expected files include:

- `tiktok_videos.csv`
  - Exported from the TikTok Display API
  - One row per public TikTok video
  - Includes fields such as:
    - video ID
    - creation time
    - caption / description
    - duration
    - views
    - likes
    - comments
    - shares

- TikTok Shop affiliate-order `.xlsx` exports
  - Contains order-level commerce information
  - Relevant fields include:
    - Content ID
    - Order ID
    - Product ID
    - Product name
    - Items sold
    - Items refunded
    - GMV
    - commission
    - order date

The pipeline joins TikTok Display API `id` with TikTok Shop `Content ID`.