# WildMatch-CLIP-LLM-Fusion

## 🎯 Overview

**WildMatch-CLIP-LLM-Fusion** is a zero-shot species classification approach that fuses:

1. **Visual scores** from CLIP (image ↔ KB text embeddings)
2. **Textual scores** from LLM descriptions (description ↔ KB embeddings)

### Score Fusion Formula

```
final_score = α × visual_score + (1 - α) × textual_score
```

Where `α` controls the balance between visual and textual evidence.

---

## 🏗️ Architecture

### Components

```
src/
├── clip_interface.py          # CLIP model interface (zero-shot)
└── matcher_clip_llm_fusion.py # Score fusion matcher

pipelines/
└── wildmatch_clip_llm_fusion.py  # Main pipeline

experiments/
└── run_clip_llm_fusion.py     # Experiment runner

config/
└── clip_llm_fusion.yaml       # Configuration
```

### Pipeline Flow

1. **VLM Description**: Generate textual description of image (LLM)
2. **CLIP Image**: Encode image with CLIP
3. **CLIP KB**: Encode KB text with CLIP (cached)
4. **Visual Score**: `cosine(clip_img, clip_kb_text)`
5. **Textual Score**: `cosine(embed(description), embed(kb_text))`
6. **Fusion**: Weighted combination of scores
7. **Prediction**: Species with highest fused score

---

## 🚀 Usage

### Interactive Demo

```bash
python main_clip_fusion.py
```

### Experiment Runner

```bash
# Run CLIP-LLM Fusion only
python experiments/run_clip_llm_fusion.py --config config/clip_llm_fusion.yaml --mode fusion

# Compare with Original WildMatch
python experiments/run_clip_llm_fusion.py --config config/clip_llm_fusion.yaml --mode comparison
```

### Pipeline Launcher

```bash
./pipeline.sh
# Select option 3: Run CLIP-LLM Fusion
# Select option 5: Compare Original vs CLIP-LLM Fusion
```

---

## ⚙️ Configuration

Edit `config/clip_llm_fusion.yaml`:

```yaml
fusion:
  alpha: 0.4              # Visual weight (0=textual only, 1=visual only)
  normalize_scores: true  # Normalize before fusion

clip:
  model: "ViT-L/14"      # CLIP model variant
  prefix: "camera trap image of an animal. "

vlm:
  model: "gpt-4o-mini"
  n_captions: 5          # Self-consistency samples
```

---

## 📊 Metrics

The experiment computes:

- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Macro-averaged metrics
- **Top-K Accuracy**: Species in top-3 and top-5 predictions

---

## 🧪 Comparison with Baselines

The system supports comparing three approaches:

| Approach | Visual Component | Textual Component | Matching |
|----------|-----------------|-------------------|----------|
| **Original WildMatch** | ❌ | LLM description | LLM reasoning |
| **Structured WildMatch** | ❌ | JSON attributes | Similarity |
| **CLIP-LLM Fusion** | ✅ CLIP | LLM description | Score fusion |

---

## 🔬 Key Features

### Zero-Shot CLIP
- No CLIP fine-tuning required
- Uses pre-trained ViT models
- KB text encoded as CLIP text (not simple templates)

### Modular Design
- Independent CLIP interface
- Reusable components
- Easy to extend

### Self-Consistency
- Multiple LLM descriptions
- Voting mechanism
- Improved robustness

### Score Normalization
- Normalizes visual and textual scores to [0,1]
- Prevents one modality from dominating
- Configurable fusion weight

---

## 📦 Installation

```bash
# Install dependencies (includes CLIP)
pip install -r requirements.txt

# Note: CLIP requires PyTorch
# GPU recommended for faster CLIP encoding
```

---

## 💡 Usage Example

```python
from pipelines.wildmatch_clip_llm_fusion import WildMatchCLIPLLMFusion
from src.utils import load_json

# Initialize
pipeline = WildMatchCLIPLLMFusion(
    openai_api_key="your-key",
    clip_model="ViT-L/14",
    alpha=0.4,
    normalize_scores=True
)

# Load KB
kb = load_json("data/knowledge_base.json")

# Predict
result = pipeline.predict(
    image_path="path/to/image.jpg",
    knowledge_base=kb,
    n_captions=5,
    verbose=True
)

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

---

## 🎯 Success Criteria

✅ **Completed:**
- Original WildMatch preserved
- CLIP-LLM Fusion implemented as independent variant
- No CLIP training (zero-shot only)
- Reuses existing textual KB
- Compatible with experiment pipeline
- Modular, documented code
- Configuration via YAML
- Comparison experiments supported

---

## 📝 Notes

- **Alpha parameter**: Start with 0.4 (slight visual bias). Tune based on dataset.
- **CLIP model**: ViT-L/14 is more accurate but slower. Use ViT-B/32 for speed.
- **KB prefix**: The prefix "camera trap image of an animal." helps CLIP understand context.
- **Caching**: CLIP and text embeddings are cached for efficiency.

---

## 🔮 Future Extensions

- [ ] Learn optimal alpha from validation set
- [ ] Ensemble multiple CLIP models
- [ ] Add attention mechanism for fusion
- [ ] Support image crops (if bounding boxes available)
- [ ] Multi-modal self-consistency (vote on fused scores)
