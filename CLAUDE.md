# Creator Intelligence — Project Context

## Project Goal

Creator Intelligence is a personalized machine learning system for TikTok Shop affiliate creators.

The system learns from a creator's historical shoppable videos and TikTok Shop performance in order to predict how a new video may perform and eventually provide personalized recommendations.

This is currently a Machine Learning course project, but the longer-term goal is to explore how it could become a SaaS product.

## Important Working Style

This project is also a learning project.

Do not make large architectural changes or rewrite major sections without explaining what is changing and why.

When implementing new functionality:

1. Prefer small, incremental changes.
2. Explain the purpose of new functions/classes/features.
3. Avoid unnecessary abstraction.
4. Do not modify unrelated code.
5. Before making a substantial change, summarize the proposed change.
6. Preserve the existing notebook workflow unless there is a strong reason to refactor it.
7. Do not expose or commit private TikTok data, API secrets, tokens, or credentials.

The user wants to understand the engineering and ML decisions rather than simply receive a finished system.

---

# Current Data

## TikTok Display API

Historical TikTok video data has been collected through TikTok OAuth and the Display API.

Approximately:

- 645 total historical videos
- fields include:
  - video ID
  - creation time
  - title/caption
  - video description
  - duration
  - view count
  - like count
  - comment count
  - share count
  - share URL

## TikTok Shop Data

Six TikTok Shop Affiliate Order exports were combined.

Approximately:

- 3,920 order rows
- 75 unique Content IDs associated with sales
- 74 of those IDs matched Display API video IDs

Orders were aggregated by Content ID and joined to TikTok videos using:

TikTok Display API `id` = TikTok Shop `Content ID`

Aggregated commerce fields include:

- unique_orders
- items_sold
- items_refunded
- total_gmv
- unique_products
- standard_commission
- shop_ads_commission
- first_sale
- last_sale

---

# Shoppable Video Labeling

Not every TikTok video is a TikTok Shop video.

The historical videos were manually reviewed and labeled.

Final dataset:

- 645 total videos
- 463 shoppable videos
- 182 non-shoppable videos

Among the 463 shoppable videos:

- 74 generated at least one sale
- 389 generated zero sales

This is the primary dataset for modeling.

---

# Existing Baseline Model

A baseline binary classification model has already been built.

## Current Target

`made_sale`

- 0 = shoppable video generated zero sales
- 1 = shoppable video generated at least one sale

## Current Input Features

- video description / caption / hashtags
- video duration

Caption text is converted using TF-IDF.

Classifier:

- Logistic Regression
- `class_weight="balanced"`

Chronological train/test split:

- approximately 370 training videos
- approximately 93 test videos

Baseline evaluation:

- ROC-AUC ≈ 0.642
- positive-class recall ≈ 0.22
- positive-class precision ≈ 0.13

This baseline should be preserved for comparison.

Do not treat accuracy as the primary metric because the dataset is imbalanced.

---

# Professor Feedback / Revised Direction

The final project should be more sophisticated than sale vs. no-sale classification.

The improved system should distinguish between low-performing and high-performing videos.

Desired future outputs include:

- low / medium / high performance tier
- estimated orders
- projected GMV or earnings
- probability of strong performance
- personalized recommendations
- explanation of why a video is predicted to perform that way

Potential ML tasks:

## Classification

Predict performance tiers such as:

- Low
- Medium
- High

Compare models such as:

- Logistic Regression
- Random Forest
- Gradient Boosting

## Regression

Predict a continuous measure such as:

- orders
- GMV
- GMV per 1,000 views

Because GMV is extremely skewed by viral videos, consider transformations such as:

`log1p(GMV)`

Potential regression models:

- Ridge Regression baseline
- Random Forest Regressor
- Gradient Boosting Regressor

Do not choose the final model without evaluating alternatives.

---

# Richer Feature Engineering

The next phase is adding richer information about each shoppable video.

## Product Features

Desired fields:

- product_brand
- product_category
- product_type
- product_price

Possible broad categories include:

- fitness_equipment
- apparel
- grooming
- skincare
- fragrance
- supplements
- food_drink
- other

Exact taxonomy may be refined after examining the data.

Product data exists from TikTok Shop order exports for the 74 videos that generated sales.

Approximately 389 shoppable zero-sale videos do not have order-derived product information and therefore require inference from captions and/or video content.

---

# Planned Content Features

Future structured video features may include:

## Content Format

Examples:

- product_demonstration
- talking_head
- workout_integration
- testimonial
- before_after
- lifestyle
- other

## Hook Type

Examples:

- problem_solution
- curiosity
- result_first
- direct_product
- testimonial
- other

## Binary Features

Potential features:

- CTA present
- creator visible
- product visible early
- product demonstrated
- before/after shown
- text overlay present

These definitions should be finalized before extracting them at scale.

---

# LLM / Multimodal Role

An LLM or multimodal model should primarily be used as a structured feature extractor and explanation layer, not as the core performance predictor.

Possible workflow:

Raw video/caption
→ multimodal or LLM feature extraction
→ structured ML features
→ traditional ML prediction
→ LLM-generated human-readable explanation/recommendation

Example extracted structure:

```json
{
  "product_category": "fitness_equipment",
  "product_type": "ab_machine",
  "product_brand": "flybird",
  "content_format": "product_demonstration",
  "hook_type": "problem_solution",
  "creator_visible": true,
  "product_visible_early": true,
  "cta_present": true
}
```

Before processing the entire dataset with an LLM, test the extraction approach on a small sample and inspect consistency.

---

# Current Work

The notebook has reached the section:

## Product Feature Engineering

A dataframe called `missing_product_df` has been created containing the shoppable videos where order-derived product information is unavailable.

Approximately:

- 74 shoppable videos have product information through orders
- 389 shoppable videos are missing product information

New columns have been created in `shop_videos_df`:

- product_category
- product_type
- product_brand

The next task is to design and test a method for filling these features.

---

# Privacy / Repository Rules

The repository is public or intended to be public.

Do NOT commit:

- raw TikTok video exports
- TikTok Shop order spreadsheets
- processed personal datasets
- API secrets
- access tokens
- client secrets
- `.env`
- personally sensitive commerce information

The repository should contain code, schemas, documentation, and examples but not the private underlying dataset.

Raw and processed data folders are excluded through `.gitignore`.

---

# Longer-Term SaaS Direction

The current project is personalized to one creator.

A production SaaS should not necessarily train an entirely separate large model from scratch for every user.

A future architecture could use:

shared/base model
+
creator-specific historical features
+
creator-specific calibration or personalization

New creators could begin with the shared model and become more personalized as historical performance data accumulates.

Manual labeling performed for this class project should eventually be replaced with automated structured extraction.