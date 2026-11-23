# Content Feature Schema (v1)

Status: **frozen for the 60-video manual pilot** (2026-11-22).
Machine-readable source of truth: [`content_feature_schema_v1.json`](content_feature_schema_v1.json). The notebook checks its fingerprint, so the field names and allowed values cannot change silently. A change means a new version (`content_v2`) with its own labels and cache.

## Purpose

Structured, **descriptive** features of each TikTok Shop video, for use as inputs to the sale/no-sale classifier (Stage 1).

- The extractor (a person, or later an LLM/multimodal model) **describes** the video. It never predicts performance.
- Every feature must be knowable **before publication**: it describes content the creator controlled when posting.
- The extractor never sees views, likes, comments, shares, orders, GMV, commission, tiers or model predictions.
- Anything visible in the video that reflects post-publication state is ignored, such as view counts in a screen recording or the like count on a stitched comment.

## Fields and allowed values

Every field allows `unclear` (or the field-specific equivalent) so that nothing has to be guessed.

| field | allowed values | definition |
|---|---|---|
| `hook_type` | `problem_question`, `bold_claim_or_result`, `product_reveal`, `action_in_progress`, `relatable_or_humor`, `reply_to_comment`, `direct_address`, `other`, `unclear` | Dominant attention device in the **first 3 seconds** (spoken, on-screen text, or visual). |
| `content_format` | `product_demo`, `workout_integration`, `talking_review`, `unboxing_first_look`, `before_after`, `skit_or_humor`, `lifestyle_montage`, `comparison`, `other`, `unclear` | Overall structure of the whole video. |
| `primary_mode` | `mostly_demonstration`, `mostly_talking`, `balanced`, `unclear` | More than 60% of runtime showing product use or action vs the creator talking; otherwise `balanced`. |
| `product_demo` | `yes`, `no`, `not_applicable`, `unclear` | The product is visibly **used for its function**; holding it up doesn't count. `not_applicable` = no product shown at all. |
| `problem_solution` | `yes`, `no`, `unclear` | A specific problem or pain point is stated, **then** the product is presented as the fix. |
| `transformation` | `none`, `claimed`, `shown`, `unclear` | Before/after or results. `claimed` = only said or written; `shown` = both states visibly shown. |
| `creator_on_camera` | `face_visible`, `body_only`, `not_visible`, `unclear` | Highest level of creator visibility at any point. |
| `product_visibility` | `not_shown`, `brief`, `prominent`, `unclear` | `brief` = in frame for under about 25% of runtime; `prominent` = in frame for most of the video. |
| `product_in_first_3s` | `yes`, `no`, `unclear` | Product visible within the first 3 seconds. |
| `cta_strength` | `none`, `soft`, `direct`, `unclear` | `soft` = "check it out", "you need this"; `direct` = explicit buying pointer ("link", "orange cart", "tap", "buy", "get yours"). Spoken, on-screen or in the caption. |
| `urgency_scarcity` | `none`, `time_limited`, `stock_limited`, `unclear` | Time pressure (sale ends, Black Friday deadline) or limited stock ("only a few left"). Pick the strongest if both. |
| `social_proof` | `none`, `creator_testimonial`, `third_party`, `unclear` | `creator_testimonial` = the creator's own experience over time; `third_party` = reviews, ratings, number sold, or other people's results. |
| `benefit_focus` | `performance_results`, `convenience_at_home`, `price_value`, `look_style`, `comfort_feel`, `scent`, `health_wellness`, `none_unclear` | The **single** main benefit pitched. |
| `humor` | `yes`, `no`, `unclear` | Intended to be funny (joke, skit, comedic exaggeration), not merely upbeat. |
| `audio_mode` | `on_camera_speech`, `voiceover`, `music_only`, `mixed`, `no_audio_or_unclear` | Dominant audio track. |
| `text_overlay` | `none`, `minimal`, `substantial`, `unclear` | `minimal` = one short label; `substantial` = headline/hook text, multiple lines, or burned-in captions. |

Audit-only fields (recorded by an automated extractor, **never** model inputs): `evidence`, `overall_confidence`, `modalities_used`, `schema_version`, `prompt_fingerprint`.

## Extraction source per field

| field | caption only | video frames | audio / transcript | full multimodal |
|---|---|---|---|---|
| hook_type | partial (reply, question captions) | yes | yes | best |
| content_format | weak | yes | yes | best |
| primary_mode | no | yes | yes | best |
| product_demo | weak | **yes** | | |
| problem_solution | partial | | **yes** | yes |
| transformation | `claimed` only | `shown` | `claimed` | yes |
| creator_on_camera | no | **yes** | | |
| product_visibility, product_in_first_3s | no | **yes** | | |
| cta_strength | yes | yes (on-screen text) | yes (spoken) | best |
| urgency_scarcity | yes | yes | yes | yes |
| social_proof | partial | | **yes** | yes |
| benefit_focus | partial | | **yes** | yes |
| humor | weak | yes | yes | best |
| audio_mode | no | | **yes** | |
| text_overlay | no | **yes** | | |

The caption alone covers only a few fields, and the caption TF-IDF already represents most of them. A meaningful test of these features needs video frames plus an audio transcript.

## Uncertainty and missing evidence

- Prefer `unclear` over a guess.
- If there is no video file, the visual fields are `unclear`. If there is no audio, `audio_mode = no_audio_or_unclear` and the audio-dependent fields use the frames and caption instead.
- An automated extractor must cite evidence (a quote or timestamp) for every non-default value. A value without evidence is changed to `unclear`.
- In the model, `unclear` is its own category. It's never imputed or silently dropped.

## Local video files

Convention: `data/raw/videos/<video_id>.mp4`, where `<video_id>` is the 19-digit TikTok Display API `id`, written in full (no scientific notation), for example `data/raw/videos/7000000000000000001.mp4` (illustrative id).

- One file per video, named only by its id. `.mp4` is the expected format.
- `data/raw/*` is gitignored, so video files are never committed.
- Files come from the creator's own downloads or data export. The pipeline does not download TikTok videos.

## Manual evaluation pilot (60 videos)

- **Sample:** 60 Shop videos from the **mature development set only** (`mature_split == "train"`). The chronological holdout is never used. The sample is stratified by product category, posting period and caption length, plus edge cases: reply videos, hashtag-only captions (a proxy for music or visual-only videos, since audio can't be inspected yet), very short and long videos, and `other`/`unknown` products.
- **No sales information** is used to choose or display videos.
- **Frozen by id** in `data/processed/content_eval_v1_sample_ids.csv`, so it's never redrawn.
- **Template:** `data/processed/content_eval_v1_labels.xlsx`, with id, post date, caption, share URL, one column per field (dropdown limited to the allowed values) and a notes column. It is written only if it doesn't already exist.
- **Consistency check for the labeler:** re-label 15 of the 60 videos about a week later, without looking at the first answers.
- **Kept separate** from the frozen product-extraction benchmark (`llm_eval_v1_sample_ids.csv`, `llm_eval_manual_labels.xlsx`, `llm_product_extractions_v1.jsonl`).

## Measuring extraction accuracy (later)

Per field, against the manual labels:
- Cohen's kappa (weighted kappa for ordered fields such as `cta_strength`, `product_visibility`, `text_overlay`, `transformation`), plus accuracy and macro-F1, and per-value precision/recall for binary fields.
- Coverage (share not `unclear`) and confident-but-wrong counts.
- Stability across two extraction runs.

**Acceptance rule:** a field becomes a model input only if kappa ≥ 0.6, coverage ≥ 80%, and its least common value appears in at least 5% of development videos. Otherwise it is merged into coarser values or dropped.

## Later modeling experiment

- **Reference:** Stage 1 (caption TF-IDF + `log1p(duration)` + `inf_product_category`, class-weighted logistic regression).
- **Stage 1 + content features:** accepted fields one-hot encoded (with `unclear` as its own level), fitted inside each expanding-window fold.
- **Comparison:** same folds, same validation rows, same metrics.
- **Pre-registered criterion:** pooled ROC-AUC up by at least 0.02, and sale recall and F1 each down by no more than 0.02.
- The chronological holdout is used once, only after the final configuration is chosen.
