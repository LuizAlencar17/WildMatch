# WildMatch: Zero-Shot Wildlife Species Classification

WildMatch is a zero-shot species classification system for wildlife camera trap images. The project implements five distinct approaches for identifying animal species without requiring labeled training data, enabling comprehensive comparison of pure LLM, vision-language fusion, and raw vision model strategies.

---

## Project Approaches

### Approach 1: WildMatch (LLM-Based) - `main.py`

**Pipeline Overview:**
The standard WildMatch pipeline uses a pure LLM-based approach with majority voting for species classification.

**How It Works:**
1. **Knowledge Base Construction**: Retrieves species information from Wikipedia using an LLM to extract key visual characteristics, habitat details, and distinguishing features
2. **Image Captioning**: Generates multiple natural language captions (default: 3) describing the image using a Vision-Language Model (VLM)
3. **LLM Matching**: For each caption, uses an LLM to compare the description against the knowledge base and predict the most likely species
4. **Majority Voting**: Aggregates predictions across all captions and selects the species with the most votes as the final prediction

**Key Components:**
- **Knowledge Base Builder**: Constructs species profiles from Wikipedia
- **VLM (Vision-Language Model)**: GPT-4o-mini for image captioning
- **LLM Matcher**: GPT-4o-mini for text-based species matching
- **Voting System**: Majority vote across multiple captions

**Execution:**
```bash
python main.py --dataset serengeti --image_type full
```

**Arguments:**
- `--dataset`: Dataset to use (`serengeti`, `wcs`, or `caltech`)
- `--image_type`: Image type (`full` or `cropped`)

---

### Approach 2: WildMatch-CLIP-LLM-Fusion - `main_clip_fusion.py`

**Pipeline Overview:**
An enhanced approach that fuses visual similarity scores from CLIP with textual LLM predictions for improved accuracy.

**How It Works:**
1. **Knowledge Base Loading**: Uses the same Wikipedia-based knowledge base as Approach 1
2. **Dual-Modal Scoring**:
   - **Visual Path (CLIP)**: Computes image-text similarity scores between the input image and species descriptions using CLIP embeddings
   - **Textual Path (LLM)**: Generates captions and uses LLM matching as in Approach 1
3. **Score Fusion**: Combines visual and textual scores using a weighted fusion:
   ```
   final_score = α × visual_score + (1 - α) × textual_score
   ```
   where α (alpha) controls the balance between visual and textual information
4. **Prediction**: Selects the species with the highest fused score

**Key Components:**
- **CLIP Model**: ViT-L/14 for visual-textual similarity
- **Alpha Parameter**: Visual weight (default: 0.7, meaning 70% visual, 30% textual)
- **Score Normalization**: Normalizes scores before fusion for balanced weighting
- **VLM + LLM**: Same as Approach 1 for the textual component

**Execution:**
```bash
python main_clip_fusion.py --dataset serengeti --image_type full
```

**Arguments:**
- `--dataset`: Dataset to use (`serengeti`, `wcs`, or `caltech`)
- `--image_type`: Image type (`full` or `cropped`)

**Default Configuration:**
- Alpha (visual weight): 0.7
- CLIP model: ViT-L/14
- Number of captions: 3
- Score normalization: Enabled

---

### Approach 3: WildMatch-BLIP-LLM-Fusion - `main_blip_fusion.py`

**Pipeline Overview:**
Similar to Approach 2, but uses BLIP (Bootstrapping Language-Image Pre-training) instead of CLIP for visual-textual fusion.

**How It Works:**
1. **Knowledge Base Loading**: Uses the same Wikipedia-based knowledge base as Approaches 1 and 2
2. **Dual-Modal Scoring**:
   - **Visual Path (BLIP)**: Computes image-text similarity scores using BLIP embeddings between the input image and species descriptions
   - **Textual Path (LLM)**: Generates captions and uses LLM matching as in Approach 1
3. **Score Fusion**: Combines visual and textual scores using weighted fusion:
   ```
   final_score = α × visual_score + (1 - α) × textual_score
   ```
   where α (alpha) controls the balance between visual and textual information
4. **Prediction**: Selects the species with the highest fused score

**Key Components:**
- **BLIP Model**: Salesforce/blip-image-captioning-large for visual-textual similarity
- **Alpha Parameter**: Visual weight (default: 0.7, meaning 70% visual, 30% textual)
- **Score Normalization**: Normalizes scores before fusion for balanced weighting
- **VLM + LLM**: Same as Approaches 1 and 2 for the textual component

**Execution:**
```bash
python main_blip_fusion.py --dataset serengeti --image_type full
```

**Arguments:**
- `--dataset`: Dataset to use (`serengeti`, `wcs`, or `caltech`)
- `--image_type`: Image type (`full` or `cropped`)

**Default Configuration:**
- Alpha (visual weight): 0.7
- BLIP model: Salesforce/blip-image-captioning-large
- Number of captions: 3
- Score normalization: Enabled

### Approach 4: WildMatch-CLIP (Raw) - `main_clip.py`

**Pipeline Overview:**
Pure CLIP visual similarity for zero-shot classification without LLM fusion.

**How It Works:**
1. **Knowledge Base Loading**: Uses the same Wikipedia-based knowledge base
2. **Visual Matching Only**: Encodes input image with CLIP, encodes all species descriptions, computes cosine similarity scores
3. **Prediction**: Selects species with highest similarity score

**Key Components:**
- **CLIP Model**: ViT-L/14 for visual-textual similarity
- **Score Normalization**: Normalizes scores to [0, 1] range
- **No API Required**: No VLM or LLM calls needed

**Execution:**
```bash
python main_clip.py --dataset serengeti --image_type full
```

**Arguments:**
- `--dataset`: Dataset to use (`serengeti`, `wcs`, or `caltech`)
- `--image_type`: Image type (`full` or `cropped`)

---

### Approach 5: WildMatch-BLIP (Raw) - `main_blip.py`

**Pipeline Overview:**
Pure BLIP visual similarity for zero-shot classification without LLM fusion.

**How It Works:**
1. **Knowledge Base Loading**: Uses the same Wikipedia-based knowledge base
2. **Visual Matching Only**: Encodes input image with BLIP, encodes all species descriptions, computes cosine similarity scores
3. **Prediction**: Selects species with highest similarity score

**Key Components:**
- **BLIP Model**: Salesforce/blip-itm-base-coco for visual-textual similarity
- **Score Normalization**: Normalizes scores to [0, 1] range
- **No API Required**: No VLM or LLM calls needed

**Execution:**
```bash
python main_blip.py --dataset serengeti --image_type full
```

**Arguments:**
- `--dataset`: Dataset to use (`serengeti`, `wcs`, or `caltech`)
- `--image_type`: Image type (`full` or `cropped`)

---

## Key Differences

| Feature | Approach 1 (LLM) | Approach 2 (CLIP-Fusion) | Approach 3 (BLIP-Fusion) | Approach 4 (CLIP-Raw) | Approach 5 (BLIP-Raw) |
|---------|------------------|--------------------------|--------------------------|------------------------|------------------------|
| **Visual Processing** | Indirect (captions) | Direct (CLIP) | Direct (BLIP) | Direct (CLIP) | Direct (BLIP) |
| **Textual Processing** | LLM matching | LLM + CLIP fusion | LLM + BLIP fusion | None | None |
| **Decision Method** | Majority voting | Weighted fusion | Weighted fusion | Direct similarity | Direct similarity |
| **Requires API Key** | Yes (OpenAI) | Yes (OpenAI) | Yes (OpenAI) | No | No |
| **Model** | GPT-4o-mini | CLIP ViT-L/14 | BLIP Salesforce | CLIP ViT-L/14 | BLIP Salesforce |
| **Speed** | Slow | Medium | Medium | Fast | Fast |
| **Strength** | Rich semantics | Visual-text balance | Captioning-focused | Pure visual baseline | Pure visual baseline |
| **Use Case** | Reference | Production | Production | Baseline comparison | Baseline comparison |

---

## Comparative Study

This project enables a comprehensive comparative study of five different zero-shot approaches:
1. **Pure LLM approach** (main.py) - Semantic text-based matching with majority voting
2. **CLIP-LLM fusion** (main_clip_fusion.py) - CLIP visual + LLM textual fusion
3. **BLIP-LLM fusion** (main_blip_fusion.py) - BLIP visual + LLM textual fusion
4. **CLIP-Raw** (main_clip.py) - Pure CLIP visual similarity baseline
5. **BLIP-Raw** (main_blip.py) - Pure BLIP visual similarity baseline

By running all five approaches on the same datasets, you can analyze:
- **Accuracy Rankings**: Compare all approaches for species classification accuracy
- **Fusion Value**: Quantify improvement gained by adding LLM to vision models
- **Raw Model Performance**: Establish pure vision model baselines
- **Speed Trade-offs**: Visual-only (fast) vs. fusion (slower) approaches
- **Cost Analysis**: API-free (Approaches 4-5) vs. API-required (Approaches 1-3)
- **Model Comparison**: CLIP vs. BLIP across different strategy types
- **Optimal Design**: Determine best approach for your specific use case

---

## Output

All five approaches generate prediction CSV files in the `results/predictions/` directory following this naming convention:
- LLM: `{dataset}_{image_type}_predictions.csv`
- CLIP-LLM Fusion: `{dataset}_{image_type}_clip_fusion_predictions.csv`
- BLIP-LLM Fusion: `{dataset}_{image_type}_blip_fusion_predictions.csv`
- CLIP-Raw: `{dataset}_{image_type}_clip_predictions.csv`
- BLIP-Raw: `{dataset}_{image_type}_blip_predictions.csv`

CSV columns (consistent across all approaches):
- `image_path`: Full path to the image
- `image_id`: Image filename
- `true_species`: Ground truth species label
- `predicted_species`: Model prediction
- `confidence`: Prediction confidence score
- `visual_scores`: Dictionary of all species scores
- `correct`: Boolean indicating if prediction matches ground truth

---

## Prerequisites

- Python 3.8+
- Required packages (see `requirements.txt`)
- For LLM and Fusion approaches: OpenAI API key (set in `.env` file as `OPENAI_API_KEY`)
- For CLIP models (Approaches 2, 4): `pip install openai-clip torch`
- For BLIP models (Approaches 3, 5): `pip install transformers torch pillow`

**Note:** Approaches 4 and 5 (raw models) do NOT require an API key, making them useful for cost-effective baseline comparisons.

---

## Project Structure

```
WildMatch/
├── main.py                          # Approach 1: LLM-based pipeline
├── main_clip_fusion.py              # Approach 2: CLIP-LLM fusion pipeline
├── main_blip_fusion.py              # Approach 3: BLIP-LLM fusion pipeline
├── main_clip.py                     # Approach 4: CLIP-Raw pipeline
├── main_blip.py                     # Approach 5: BLIP-Raw pipeline
├── src/                             # Core modules
│   ├── knowledge_base.py            # Wikipedia-based KB construction
│   ├── pipeline.py                  # LLM-based prediction
│   ├── matcher_clip_llm_fusion.py   # CLIP-LLM fusion matcher
│   ├── matcher_blip_llm_fusion.py   # BLIP-LLM fusion matcher
│   ├── matcher_clip.py              # CLIP-Raw matcher
│   ├── matcher_blip.py              # BLIP-Raw matcher
│   ├── clip_interface.py            # CLIP model interface
│   ├── blip_interface.py            # BLIP model interface
│   ├── vlm.py                       # Vision-Language Model interface
│   └── utils.py                     # Utilities
├── pipelines/                       # Pipeline implementations
│   ├── wildmatch_clip_llm_fusion.py # CLIP-LLM fusion pipeline
│   ├── wildmatch_blip_llm_fusion.py # BLIP-LLM fusion pipeline
│   ├── wildmatch_clip.py            # CLIP-Raw pipeline
│   └── wildmatch_blip.py            # BLIP-Raw pipeline
├── data/                            # Datasets and knowledge bases
│   ├── serengeti/
│   ├── wcs/
│   └── caltech/
└── results/                         # Prediction outputs
    └── predictions/
```

---

## Citation

If you use this code, please reference the WildMatch project and its methodology for zero-shot wildlife species classification.
