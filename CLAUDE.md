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

# Current State (checkpoint 2026-11-23)

The counts in "Current Data", "Shoppable Video Labeling" and "Existing Baseline Model" above describe the original 645-video snapshot. They are superseded by this section.

## Dataset state

- Refreshed Display API pull: **700 videos**. Order exports are deduplicated on `Order ID + SKU ID`, keeping the newest export.
- **516** classified as Shop videos (manual labels plus `MANUAL_LABEL_CORRECTIONS`).
- **465** mature Shop videos under the **30-day maturity rule** (a video counts only once 30 days have passed since posting).
- Mature chronological split (`mature_split`): **372 development / 93 holdout**.
- **The holdout must stay untouched** unless the user explicitly approves a final evaluation. It has been evaluated exactly once, for Model B.
- Modeling export: `data/processed/shop_videos_modeling.csv` plus a column-role sidecar JSON. Inference features are prefixed `inf_*`, ground-truth-derived ones `gt_*`. Only `inf_*` features and pre-publication fields are model inputs.
- Targets: `performance_tier` (0 orders = low, 1 = medium, 2+ = high) and `target_log1p_orders`.
- Development evaluation uses **3 expanding-window folds** (validation blocks: Oct–Nov 2025, Dec 2025, Jan–May 2026).

## Modeling conclusions so far

- The 3-class low/medium/high formulation is weak.
- **Model B** (caption TF-IDF + `log1p(duration)` + `inf_product_category`, class-weighted logistic regression) was the best first 3-class model. It only weakly beat trivial baselines and missed the high tier on the holdout.
- **Two-stage experiment:** Stage 1 (sale vs no-sale) holds the only meaningful signal so far.
  - **Stage 1 reference** = Model B's inputs, class-weighted logistic regression.
  - Expanding-window pooled ROC-AUC is **~0.63**.
  - Stage 2 (medium vs high among sales) shows no reliable ranking signal (AUC ~0.48).
- Ridge regression on `log1p(orders)` did not beat simple baselines and showed little useful signal about order magnitude.
- Caption-metadata features (B+) and simple seasonality flags did not meaningfully improve Stage 1.
- Experiments use criteria fixed before fitting. For Stage 1 the bar is: pooled ROC-AUC up by at least 0.02, with sale recall and sale F1 each down by no more than 0.02.

## Product feature pipeline (frozen)

Product brand/type/category come from this hierarchy, in order:
1. order-backed product data;
2. manual overrides;
3. deterministic rules (frozen v1 extractor);
4. cached LLM fallback (`llm_product_extractions_v1.jsonl`, caption only, prompt fingerprint guarded);
5. `unknown`.

The 64-video product benchmark (frozen by ID) and its cache stay frozen unless explicitly versioned. `RUN_LLM = False`.

## Multimodal content-feature experiment (v1)

- **Schema:** frozen v1 with 16 fields, in `docs/content_feature_schema_v1.json` (fingerprint-checked) and `docs/content_feature_schema.md`.
- **Benchmark:** 60 mature development videos were manually labeled.
  - Frozen IDs: `data/processed/content_eval_v1_sample_ids.csv`.
  - Labels: `data/processed/content_eval_v1_labels.xlsx`.
  - Both are private and gitignored.
- **Video files:** the local MP4s are in `data/raw/videos/<id>.mp4`. **They are private and must never be committed.**
- **Local preparation** (no API):
  - Frames are sampled with OpenCV: 7 frames over the first 3 s, then up to 8 spread over the rest, resized to 1024 px.
  - Audio is decoded with PyAV and transcribed locally with faster-whisper `small`.
  - Loudness and non-silent-time statistics are computed locally.
  - Transcripts are cached in `data/processed/content_transcripts_v1.jsonl` (private).
  - faster-whisper's own `decode_audio` is incompatible with PyAV 19, so the notebook uses `decode_audio_16k`.
- **Extractor request:** caption, sampled frames, local transcript and audio statistics. Model `claude-opus-5-5`, effort `medium`, structured outputs.
  - It **never** receives sales, orders, GMV, engagement, targets, manual benchmark labels or model predictions. The video ID isn't sent either.
- **Cache:** `data/processed/content_extractions_content_v1_p1.jsonl`, append-only. Prompt fingerprint `92e5919f…`.
  - It holds 60 first-pass records (replicate 1) and 15 stability records (replicate 2).
  - `MAX_CONTENT_CALLS = 75` is now fully used.
  - Measured cost is about $0.08 per video (about 16k input / 800 output tokens).
- **Notebook switches** (all default False): `PREPARE_CONTENT_MEDIA`, `RUN_CONTENT_EXTRACTOR`, `RUN_CONTENT_SMOKE_TEST`, `SCORE_CONTENT_EXTRACTIONS`.

## v1 extractor evaluation

- **Agreement with the human labels:** 676/960 values = **70.4%** exact match.
- **Stability:** a 15-video replicate run agreed with itself on 229/240 values = **95.4%**.
- High stability with lower human agreement means the gaps are mostly **schema-definition disagreements**, not random extractor behavior.
- **Approved for modeling:** `product_demo` (κ 0.82), `product_in_first_3s` (κ 0.76), `problem_solution` (κ 0.71).
  - All three passed the pre-registered criteria: κ ≥ 0.60, coverage ≥ 80%, rarest value ≥ 5%. All three are highly stable.
- **`audio_mode`:** fairly accurate (κ 0.62) and 100% stable, but needs value merging or local music detection before modeling.
- **Do not use the other fields** without a v2 redesign.
- **Do not tune v1 definitions** to this same 60-video benchmark and then report the new score as an unbiased evaluation. A v2 needs re-labeling under the new rules and a fresh labeled sample.

## v1 limitations / future work (v2)

- **content_format:** fuzzy boundaries, especially workout vs product demonstration for fitness equipment.
- **hook_type:** `other` is too broad (half the human labels).
- **social_proof:** the human labels count personal endorsement as a testimonial; the extractor reads "experience over time" strictly.
- **creator_on_camera:** unclear whether hands count as "body".
- **text_overlay:** the minimal vs substantial boundary is ambiguous (one persistent headline).
- **cta_strength:** needs explicit rules (e.g. "link down below").
- **product_visibility:** not useful enough (κ −0.10; screen time can't be judged from still frames).
- **humor and benefit_focus:** too sparse or fragmented. `benefit_focus` largely repeats product category.
- **audio_mode:** can't tell voiceover from speech over music with the current inputs.
- **primary_mode:** has no value for "product displayed, nothing happening".

## Exact next step when resuming (NOT started)

Test whether the three validated semantic features improve Stage 1 sale-vs-no-sale prediction.

Before any new API call or model fit:
1. Inspect the current modeling dataset and the content-extraction cache.
2. Count how many mature, development and holdout videos already have extractor outputs.
3. Count how many more videos would need extraction.
   - The v1 cache cap (75) is used up. Any further calls need an explicit, approved cap change under the same frozen prompt fingerprint.
4. Estimate the API cost from actual v1 usage (~$0.08 per video).
5. Plan a development-only ablation:
   - Stage 1 reference;
   - + `product_demo`;
   - + `product_demo` and `product_in_first_3s`;
   - + all three approved semantic features.
6. Use **extractor outputs** as features, NEVER the human benchmark labels.
7. Keep the holdout untouched.
8. Define the improvement criterion before training.
9. Stop after the plan, counts and cost estimate, and wait for approval.

## Notebook-run conventions

- Notebook outputs are stripped on commit by the local `nbstripout` git filter, set in `.git/info/attributes`. It isn't tracked, so re-run `nbstripout --install` in a fresh clone.
- All API switches stay False in the committed notebook. API calls happen only after explicit user approval, with a capped number of calls and retries disabled.

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