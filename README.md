# Creator Intelligence

Creator Intelligence is a machine-learning project designed to learn what kinds of TikTok Shop videos perform best for an individual creator.

## Problem

Existing creator analytics tools provide general trends and aggregate performance data, but creators may benefit from recommendations based specifically on their own historical content and sales performance.

## Current Pipeline

1. Authenticate with TikTok using OAuth.
2. Retrieve historical video metadata through the TikTok Display API.
3. Import TikTok Shop affiliate-order exports.
4. Aggregate order performance by video.
5. Join video and commerce data using TikTok video ID / Content ID.
6. Identify which historical videos are shoppable.
7. Engineer creator-specific features.
8. Train models to estimate future commercial performance.

## Current Baseline

The current baseline predicts whether a shoppable video will generate at least one sale using:

- caption / hashtags
- video duration
- TF-IDF text features
- Logistic Regression

The baseline is intentionally simple and will be expanded with richer product and video-content features.

## Planned Improvements

- low / medium / high performance prediction
- projected GMV or earnings
- product/category features
- hook and CTA detection
- video-format features
- multimodal / LLM-assisted feature extraction
- personalized creator recommendations