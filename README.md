# WildMatch: Zero-Shot Wildlife Species Classification

WildMatch is a zero-shot species classification system for wildlife camera trap images. The project implements two distinct approaches for identifying animal species without requiring labeled training data.

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

---

## Key Differences

| Feature | Approach 1 (LLM-Based) | Approach 2 (CLIP-LLM Fusion) | Approach 3 (BLIP-LLM Fusion) |
|---------|------------------------|------------------------------|------------------------------|
| **Visual Processing** | Indirect (via captions) | Direct (CLIP embeddings) | Direct (BLIP embeddings) |
| **Textual Processing** | LLM matching with voting | LLM matching with fusion | LLM matching with fusion |
| **Decision Method** | Majority voting | Weighted score fusion | Weighted score fusion |
| **Visual Model** | N/A | OpenAI CLIP (ViT-L/14) | Salesforce BLIP (large) |
| **Model Training** | N/A | Contrastive learning | Bootstrapped captioning |
| **Primary Strength** | Rich semantic understanding | Strong zero-shot visual-text alignment | Optimized for image captioning tasks |
| **Complexity** | Simpler, LLM-only | Multi-modal fusion | Multi-modal fusion |

---

## Comparative Study

This project enables a comparative study of three different zero-shot approaches:
1. **Pure LLM approach** (main.py) - Semantic text-based matching
2. **CLIP-LLM fusion** (main_clip_fusion.py) - CLIP visual embeddings + LLM
3. **BLIP-LLM fusion** (main_blip_fusion.py) - BLIP visual embeddings + LLM

By running all three approaches on the same datasets, you can compare:
- **Accuracy**: Which approach achieves better species classification?
- **Confidence**: How confident is each model in its predictions?
- **Speed**: How do inference times compare?
- **Visual vs. Textual**: What's the optimal balance (alpha) for each vision model?

---

## Output

All three approaches generate prediction CSV files in the `results/predictions/` directory with the following columns:
- `image_path`: Full path to the image
- `image_id`: Image filename
- `true_species`: Ground truth species label
- `predicted_species`: Model prediction
- `confidence`: Prediction confidence score
- `vote_counts`: Vote distribution (Approach 1) or score details (Approach 2)
- `correct`: Boolean indicating if prediction matches ground truth

---

## Prerequisites

- Python 3.8+
- OpenAI API key (set in `.env` file as `OPENAI_API_KEY`)
- Required packages (see `requirements.txt`)
- For CLIP fusion: `pip install clip torch`
- For BLIP fusion: `pip install transformers torch pillow`

---

## Project Structure

```
WildMatch/
├── main_blip_fusion.py              # Approach 3: BLIP-LLM fusion pipeline
├── src/                             # Core modules
│   ├── knowledge_base.py            # Wikipedia-based KB construction
│   ├── pipeline.py                  # LLM-based prediction
│   ├── matcher_clip_llm_fusion.py   # CLIP-LLM fusion matcher
│   ├── matcher_blip_llm_fusion.py   # BLIP-LLM fusion matcher
│   ├── clip_interface.py            # CLIP model interface
│   ├── blip_interface.py            # BLIP model interface
│   ├── vlm.py                       # Vision-Language Model interface
│   └── utils.py                     # Utilities
├── pipelines/                       # Pipeline implementations
│   ├── wildmatch_clip_llm_fusion.py # CLIP fusion pipeline
│   └── wildmatch_blip_llm_fusion.py # BLIP ftilities
├── pipelines/                       # Pipeline implementations
│   └── wildmatch_clip_llm_fusion.py # Fusion pipeline
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
