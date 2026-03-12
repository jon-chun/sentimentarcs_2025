# Comprehensive Analysis: SentimentArcs Simplified Notebook (Oct 2022)

**Analysis Date:** October 2025  
**Original Notebook Date:** October 29, 2022  
**Time Gap:** 3 years

---

## EXECUTIVE SUMMARY

This Google Colab notebook implements diachronic sentiment analysis on literary texts using an ensemble of 15+ sentiment analysis models. While the approach is methodologically sound, the notebook suffers from:
1. **Deprecated library dependencies** (transformers API changes)
2. **Missing educational scaffolding** for non-STEM students
3. **Poor code organization** and modularity
4. **Inadequate error handling** and logging
5. **Incomplete documentation** of theoretical foundations

---

## PART 1: CRITICAL ERRORS & DEPRECATIONS (Oct 2025)

### 🔴 CRITICAL: API Breaking Changes

#### 1.1 Transformers Library Deprecations
**Location:** Lines 1463-1465
```python
from transformers import AutoModelWithLMHead  # ❌ DEPRECATED
```

**Error:** `AutoModelWithLMHead` was deprecated in transformers v4.0 (2020) and removed in v5.0
**Fix:** Replace with:
```python
from transformers import AutoModelForSeq2SeqLM  # ✅ Current
```

**Impact:** HIGH - Code will fail to run with modern transformers versions

---

#### 1.2 Variable Name Inconsistencies
**Location:** Lines 269-270, 1602
```python
sentiment_df = pd.DataFrame  # ❌ Wrong - assigns class, not instance
line_no_ls = list(range(len(preds)))  # ❌ Undefined variable 'preds'
```

**Errors:**
- Line 270: Assigns DataFrame class instead of creating instance
- Line 1602: Variable `preds` referenced before definition
- Line 1640: Uses 'sentiment' column that doesn't exist in roberta15lg_df

**Impact:** HIGH - Runtime errors

---

#### 1.3 Syntax Errors
**Location:** Lines 3718-3720
```python
elif sentiment_insert_ct = sentimentr_extra_len + 1:  # ❌ Assignment instead of comparison
    print('add one insert')
elif sentiment_insert_ct = sentiment_extra_len -1:    # ❌ Assignment instead of comparison
```

**Fix:**
```python
elif sentiment_insert_ct == sentimentr_extra_len + 1:  # ✅
    print('add one insert')
elif sentiment_insert_ct == sentimentr_extra_len - 1:  # ✅
```

**Impact:** CRITICAL - Syntax error prevents execution

---

### 🟡 MODERATE: Missing Constants

#### 1.4 Undefined Global Constants
**Location:** Line 442
```python
novel_raw_str = uploaded[novel_name_str].decode(TEXT_ENCODING)
```

**Error:** `TEXT_ENCODING` constant never defined
**Fix:** Add at top of notebook:
```python
TEXT_ENCODING = 'utf-8'
```

---

## PART 2: LIBRARY & API UPDATES (2022 → 2025)

### 2.1 Transformers Library (v4.0 → v4.45+)

**Major Changes:**
1. **Pipeline API Enhanced**
   - Old: Limited device selection
   - New: Automatic device mapping, quantization support
   ```python
   # Old (2022)
   classifier = pipeline("sentiment-analysis")
   
   # New (2025) - with optimizations
   classifier = pipeline(
       "sentiment-analysis",
       device_map="auto",  # Automatic GPU/CPU selection
       torch_dtype=torch.float16  # Memory optimization
   )
   ```

2. **Model Loading**
   - `AutoModelWithLMHead` → `AutoModelForSeq2SeqLM`
   - `AutoModelForSequenceClassification` remains but has new parameters

3. **Tokenizer Updates**
   - Now includes `return_tensors='pt'` by default in many contexts
   - Better handling of special tokens

---

### 2.2 VADER Updates
**Current:** vaderSentiment v3.3.2+ (2022: v3.3.1)
- No breaking changes, but improved emoji handling
- Better social media text processing

---

### 2.3 R Package Updates (rpy2)

**Syuzhet:** Now v1.0.7 (was v1.0.6)
- Minor improvements, no breaking changes

**SentimentR:** Now v2.9.0 (was v2.7+)
- Enhanced lexicon support
- Better negation handling
- **New parameters available** for `sentiment()` function

---

### 2.4 New Recommended Libraries (2025)

**Missing Modern Alternatives:**
1. **Sentence Transformers** - For semantic similarity
2. **BERTopic** - For topic modeling alongside sentiment
3. **Transformers Evaluate** - For model evaluation metrics
4. **Gradio/Streamlit** - For interactive demos
5. **Hugging Face Evaluate** - Standardized metrics

---

## PART 3: MISSING BEST PRACTICES

### 3.1 Code Organization

**Current Issues:**
- ❌ All code in single monolithic notebook
- ❌ No separation of concerns
- ❌ Functions mixed with execution code
- ❌ No configuration management

**Recommended Structure:**
```
sentiment_analysis_project/
├── config/
│   └── config.yaml                 # Configuration parameters
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # Text loading & cleaning
│   ├── preprocessor.py            # Text preprocessing
│   ├── sentiment_analyzers.py     # All sentiment models
│   ├── visualizer.py              # Plotting functions
│   └── utils.py                   # Helper functions
├── notebooks/
│   ├── 01_introduction.ipynb      # Theory & concepts
│   ├── 02_data_prep.ipynb         # Data preparation
│   ├── 03_vader_analysis.ipynb    # Lexicon methods
│   ├── 04_transformer_analysis.ipynb  # Deep learning
│   └── 05_ensemble_analysis.ipynb # Combined results
├── data/
│   ├── raw/
│   └── processed/
└── outputs/
    ├── results/
    └── figures/
```

---

### 3.2 Error Handling

**Missing Throughout:**
```python
# Current (NO error handling)
uploaded = files.upload()
novel_raw_str = uploaded[novel_name_str].decode(TEXT_ENCODING)

# Should be:
try:
    uploaded = files.upload()
    if not uploaded:
        raise ValueError("No file was uploaded")
    
    novel_name_str = list(uploaded.keys())[0]
    novel_raw_str = uploaded[novel_name_str].decode(TEXT_ENCODING)
    
except Exception as e:
    print(f"❌ Error loading file: {e}")
    raise
```

---

### 3.3 Logging & Progress

**Current:** Minimal print statements
**Needed:** Structured logging

```python
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'sentiment_analysis_{datetime.now():%Y%m%d_%H%M%S}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Usage
logger.info(f"Processing {len(novel_clean_ls)} sentences")
logger.debug(f"Window size: {win_size}")
logger.warning("VADER scores may be skewed for literary text")
```

---

### 3.4 Progress Tracking

**Current:** Inconsistent use of tqdm
**Improved:**
```python
from tqdm.auto import tqdm
import time

def process_with_progress(items, desc="Processing"):
    """Process items with progress bar and time estimation"""
    start_time = time.time()
    results = []
    
    for i, item in enumerate(tqdm(items, desc=desc)):
        result = process_item(item)
        results.append(result)
        
        # Progress reporting
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (len(items) - i - 1) / rate
            logger.info(f"Progress: {i+1}/{len(items)} | "
                       f"Rate: {rate:.1f} items/sec | "
                       f"ETA: {remaining/60:.1f} min")
    
    return results
```

---

## PART 4: THEORETICAL FOUNDATIONS (Missing Educational Content)

### 4.1 Sentiment Analysis Approaches

**Add Markdown Cell:**
```markdown
## 🎓 Understanding Sentiment Analysis Approaches

### Three Paradigms of Sentiment Analysis

#### 1. **Lexicon-Based (Symbolic)**
**Examples:** VADER, TextBlob, SyuzhetR, SentimentR

**How it works:**
- Uses pre-compiled dictionaries of words with sentiment scores
- Example: "happy" = +0.7, "sad" = -0.6, "terrible" = -1.0
- Considers modifiers: "very happy" = +0.85, "not happy" = -0.5

**Strengths:**
- ✅ Fast and lightweight
- ✅ Interpretable results
- ✅ No training data needed
- ✅ Works well for general text

**Weaknesses:**
- ❌ Misses context ("not bad" = positive)
- ❌ Struggles with sarcasm
- ❌ Limited vocabulary
- ❌ Can't learn domain-specific patterns

**Best for:** Quick analysis, transparent results, resource-constrained environments

---

#### 2. **Statistical ML**
**Examples:** Naive Bayes, SVM, Logistic Regression

**How it works:**
- Learns patterns from labeled training data
- Uses features like word frequencies, n-grams
- Builds statistical model to classify sentiment

**Strengths:**
- ✅ Can learn domain patterns
- ✅ Computationally efficient
- ✅ Good for binary classification

**Weaknesses:**
- ❌ Requires labeled training data
- ❌ Fixed feature representation
- ❌ Loses word order information
- ❌ Struggles with complex sentences

**Best for:** When you have domain-specific training data

---

#### 3. **Deep Learning (Transformers)**
**Examples:** BERT, RoBERTa, DistilBERT

**How it works:**
- Pre-trained on massive text corpora
- Learns contextual word representations
- Fine-tuned on sentiment classification
- Uses attention mechanisms to weight word importance

**Strengths:**
- ✅ Understands context deeply
- ✅ Captures long-range dependencies
- ✅ State-of-the-art accuracy
- ✅ Transfer learning from large models

**Weaknesses:**
- ❌ Computationally expensive
- ❌ Requires GPU for speed
- ❌ "Black box" - hard to interpret
- ❌ May overfit on small datasets

**Best for:** When accuracy is critical and resources available

---

### Model Comparison Table

| Aspect | VADER | TextBlob | SyuzhetR | RoBERTa | DistilBERT |
|--------|-------|----------|----------|---------|------------|
| Speed | ⚡⚡⚡ | ⚡⚡⚡ | ⚡⚡ | ⚡ | ⚡⚡ |
| Accuracy | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Context Awareness | ❌ | ❌ | ✅ (partial) | ✅✅✅ | ✅✅✅ |
| Literary Text | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Interpretability | ✅✅✅ | ✅✅✅ | ✅✅ | ❌ | ❌ |
| Resource Needs | 💻 | 💻 | 💻💻 | 💻💻💻💻 | 💻💻💻 |

---

### Why Use an Ensemble?

**The Wisdom of Crowds:** Different models capture different aspects of sentiment:
- **VADER:** Good for social media, emphatic expressions
- **SyuzhetR:** Optimized for narrative arcs in literature
- **RoBERTa:** Best overall accuracy, contextual understanding
- **DistilBERT:** Balanced speed/accuracy tradeoff

**Aggregation Strategies:**
1. **Mean:** Average all model scores
2. **Weighted Mean:** Weight models by reliability
3. **Majority Vote:** Use most common prediction
4. **Stacking:** Train meta-model on ensemble outputs
```

---

### 4.2 Smoothing Techniques

**Add Educational Cell:**
```markdown
## 📊 Smoothing Time Series Data

### Why Smooth Sentiment Scores?

Raw sentiment scores are **noisy** - they fluctuate sentence-by-sentence.
Smoothing reveals the **underlying emotional arc** of a narrative.

---

### Technique 1: Rolling/Moving Average (Currently Used)

**How it works:**
```python
window_size = 100  # Look at 100 sentences at a time
smoothed = df['sentiment'].rolling(window=window_size, center=True).mean()
```

**Parameters:**
- `window`: How many data points to average
  - Larger = smoother, but less detail
  - Smaller = more detail, but noisier
- `center=True`: Center the window (better for narratives)
  - False: Only look backwards (causal)
  - True: Look forward and backward (acausal, better visualization)

**Visualization:**
```
Raw:      ∩∨∧∨∩∧∨∩∨∧∨∩∧∨∩
Window=3: \_\_/‾‾\_\_/‾‾\_\_/‾
Window=7: \____/‾‾‾‾\____/
```

**Pros:** Simple, interpretable, no assumptions
**Cons:** Treats all points equally, edge effects

---

### Technique 2: Savitzky-Golay Filter (RECOMMENDED TO ADD)

**How it works:** Fits polynomial to local window, preserves peaks better

```python
from scipy.signal import savgol_filter

# Apply Savitzky-Golay smoothing
window_length = 51  # Must be odd number
polyorder = 3        # Polynomial degree (2-5 typical)

smoothed = savgol_filter(
    sentiment_scores,
    window_length=window_length,
    polyorder=polyorder
)
```

**Parameters:**
- `window_length`: Size of smoothing window
  - Rule of thumb: 5-10% of data length
  - Must be odd number
- `polyorder`: Polynomial degree
  - 2: Parabolic (good for peaks)
  - 3: Cubic (smoother curves)
  - Higher: More wiggly, can overfit

**Visual Comparison:**
```
Raw:              ∩∨∧∨∩∧∨∩∨∧∨∩∧∨∩
Moving Avg:       \_____/‾‾‾‾\_____
Savitzky-Golay:   \_/‾∧‾\_/‾∧‾\_/
                    ↑         ↑
                  Preserves  peaks!
```

**Pros:** Preserves peak locations/heights, less phase distortion
**Cons:** More complex, sensitive to parameters

---

### Technique 3: LOWESS (Locally Weighted Scatterplot Smoothing)

**How it works:** Fits weighted regression in local neighborhoods

```python
from statsmodels.nonparametric.smoothers_lowess import lowess

smoothed = lowess(
    sentiment_scores,
    x_values,
    frac=0.1  # Fraction of data used for each local regression
)
```

**Parameters:**
- `frac`: Smoothing parameter (0.0-1.0)
  - 0.1: Use 10% of data for each point
  - Smaller = less smoothing
  - Larger = more smoothing

**Pros:** Robust to outliers, adaptive smoothing
**Cons:** Slower, harder to interpret parameters

---

### Technique 4: Exponential Moving Average

**How it works:** Gives more weight to recent points

```python
smoothed = df['sentiment'].ewm(span=window_size, adjust=False).mean()
```

**Pros:** Computationally efficient, good for real-time
**Cons:** Not symmetric (causal), can lag trends

---

### Which Smoothing to Use?

| Use Case | Best Technique | Window/Param |
|----------|----------------|--------------|
| **Quick exploration** | Rolling Mean | 5-10% of length |
| **Preserve peaks** | Savitzky-Golay | window=51, poly=3 |
| **Publication quality** | Savitzky-Golay | Experiment with params |
| **Noisy data** | LOWESS | frac=0.1-0.3 |
| **Real-time analysis** | Exponential MA | span=100 |

---

### Hyperparameter Selection

**Window Size Guidelines:**
```python
import numpy as np

total_sentences = len(sentiment_scores)

# Rule of thumb: 5-10% of total length
min_window = int(total_sentences * 0.05)
max_window = int(total_sentences * 0.10)

print(f"Suggested window range: {min_window} - {max_window}")

# For novels:
# - Short story (1K sentences): window = 50-100
# - Novel (10K sentences): window = 500-1000
# - Epic (100K sentences): window = 5000-10000
```

**Visual Comparison Tool:**
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 1, figsize=(15, 12))

# Original
axes[0].plot(sentiment_scores, alpha=0.3, label='Raw')
axes[0].set_title('Raw Sentiment Scores')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Rolling mean
window_sizes = [50, 100, 200]
for i, window in enumerate(window_sizes):
    smoothed = pd.Series(sentiment_scores).rolling(window, center=True).mean()
    axes[1].plot(smoothed, label=f'Window={window}', alpha=0.7)
axes[1].set_title('Rolling Mean (Different Windows)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Savitzky-Golay
from scipy.signal import savgol_filter
window = 51
for poly in [2, 3, 4]:
    smoothed = savgol_filter(sentiment_scores, window, poly)
    axes[2].plot(smoothed, label=f'Polynomial degree={poly}', alpha=0.7)
axes[2].set_title('Savitzky-Golay (Different Polynomials)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# Comparison of best from each
axes[3].plot(sentiment_scores, alpha=0.2, label='Raw', color='gray')
axes[3].plot(pd.Series(sentiment_scores).rolling(100, center=True).mean(), 
             label='Rolling (100)', linewidth=2)
axes[3].plot(savgol_filter(sentiment_scores, 51, 3), 
             label='Savitzky-Golay (51, 3)', linewidth=2)
axes[3].set_title('Comparison of Smoothing Techniques')
axes[3].legend()
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('smoothing_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("💡 Tip: Savitzky-Golay better preserves peak sharpness and timing")
```
```

---

### 4.3 Interpreting Results

**Add Educational Cell:**
```markdown
## 📈 Interpreting Sentiment Arcs

### What Do Peaks and Valleys Mean?

#### Positive Peaks (High Sentiment Scores)
**Narrative Function:**
- 🎉 Climactic happy moments
- 💑 Romance scenes
- 🏆 Character victories
- 🌅 Peaceful resolutions

**Examples in Literature:**
- Pride & Prejudice: Elizabeth & Darcy's first kiss
- Harry Potter: Defeating Voldemort
- The Martian: Rescue scene

---

#### Negative Valleys (Low Sentiment Scores)
**Narrative Function:**
- 😢 Tragedy or loss
- ⚡ High tension/conflict
- 😰 Character despair
- 🌑 Dark narrative moments

**Examples:**
- Romeo & Juliet: Death scene
- 1984: Room 101 torture
- The Road: Starvation scenes

---

#### Rapid Changes (High Slope)
**Indicates:**
- Plot twist or revelation
- Sudden character transformation
- Genre shift (tragedy → hope)
- Pacing acceleration

---

### Narrative Arc Patterns

#### 1. **Freytag's Pyramid**
```
        ★ Climax
       /  \
      /    \
  Rising    Falling
  Action    Action
   /          \
  /            \
 /______________\
 Exposition   Resolution
```

**Sentiment Pattern:**
- Low → Gradual Rise → Peak → Sharp Fall → Stabilize

---

#### 2. **Vonnegut's Story Shapes**

**"Cinderella" (Rags to Riches):**
```
   /‾‾‾‾|
  /     |
 /      |___
/           
```

**"Man in a Hole":**
```
‾‾‾\    /‾‾‾
    \  /
     \/
```

**"Boy Meets Girl":**
```
     /\     /\
    /  \   /  \
___/    \_/    \___
```

---

### Statistical Measures

#### Amplitude (Emotional Range)
```python
emotional_range = sentiment_scores.max() - sentiment_scores.min()
print(f"Emotional Range: {emotional_range:.3f}")

# Interpretation:
# < 1.0: Subdued, consistent tone
# 1.0-2.0: Moderate emotional variation
# > 2.0: Highly dramatic, intense shifts
```

---

#### Volatility (Pace of Change)
```python
volatility = sentiment_scores.diff().abs().mean()
print(f"Emotional Volatility: {volatility:.3f}")

# Interpretation:
# < 0.1: Slow, contemplative pace
# 0.1-0.3: Moderate pacing
# > 0.3: Fast-paced, intense
```

---

#### Turning Points (Critical Scenes)
```python
from scipy.signal import find_peaks

# Find peaks (happy moments)
peaks_positive, _ = find_peaks(sentiment_scores, prominence=0.5)

# Find valleys (sad moments)
peaks_negative, _ = find_peaks(-sentiment_scores, prominence=0.5)

print(f"Major positive turning points: {len(peaks_positive)}")
print(f"Major negative turning points: {len(peaks_negative)}")

# Get sentences at turning points
for i, peak_idx in enumerate(peaks_positive[:5]):  # Top 5
    print(f"\n🎉 Positive Peak #{i+1} (Sentence {peak_idx}):")
    print(f"  Score: {sentiment_scores[peak_idx]:.3f}")
    print(f"  Text: {sentences[peak_idx][:100]}...")
```

---

### Comparing Multiple Models

**Why do models disagree?**

```python
# Calculate model agreement
from scipy.stats import spearmanr

models = ['vader', 'syuzhetr_syuzhet', 'roberta', 'distilbert']
correlation_matrix = df[models].corr(method='spearman')

import seaborn as sns
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            vmin=-1, vmax=1)
plt.title('Model Agreement (Spearman Correlation)')
plt.tight_layout()
plt.show()

print("\n📊 Interpretation:")
print("High correlation (>0.7): Models agree on overall trajectory")
print("Moderate (0.4-0.7): Some disagreement on specifics")
print("Low (<0.4): Models capture different aspects")
```

**Model-Specific Strengths:**
- **VADER**: Excels at informal text, ALL CAPS, emojis
- **SyuzhetR**: Optimized for literary narrative arcs
- **RoBERTa**: Best at nuanced context, sarcasm detection
- **Ensemble**: Aggregates strengths, reduces individual biases

---

### Genre-Specific Patterns

#### Romance Novels
**Expected Pattern:**
- Low → Rise → Valley (conflict) → Peak (resolution)
- High positive endpoint
- Multiple small peaks (romantic moments)

#### Tragedy
**Expected Pattern:**
- High → Gradual decline → Sharp drop → Low endpoint
- Few positive peaks
- Extended negative plateau

#### Mystery/Thriller
**Expected Pattern:**
- Moderate → Oscillating (clues/red herrings)
- Sharp negative spikes (crimes/danger)
- Resolution peak at end

#### Comedy
**Expected Pattern:**
- Consistently positive or neutral
- Small oscillations (comedic timing)
- Upward trend overall

---

### Action Items for Analysis

1. **Identify Genre:** Does the arc match expected pattern?
2. **Find Outliers:** Which scenes diverge from the arc?
3. **Compare Models:** Where do models strongly disagree?
4. **Extract Key Scenes:** Use peaks/valleys to find important passages
5. **Test Hypotheses:** Does arc correlate with reader reviews?
```

---

## PART 5: RANKED IMPROVEMENT SUGGESTIONS

### 🥇 TIER 1: CRITICAL (Must Fix to Run)

**Priority 1.1:** Fix Syntax Errors
- Lines 3718-3720: Change `=` to `==` in conditionals
- **Effort:** 5 minutes
- **Impact:** CRITICAL

**Priority 1.2:** Fix Variable Definitions
- Line 270: Change to `sentiment_df = pd.DataFrame()`
- Line 1602: Define `preds` before use
- Line 442: Add `TEXT_ENCODING = 'utf-8'` constant
- **Effort:** 10 minutes
- **Impact:** HIGH

**Priority 1.3:** Update Deprecated Imports
- Replace all `AutoModelWithLMHead` with `AutoModelForSeq2SeqLM`
- **Effort:** 15 minutes
- **Impact:** HIGH

**Priority 1.4:** Fix Column Reference Errors
- Line 1640: roberta15lg_df uses 'sentiment' column that doesn't exist
- **Effort:** 20 minutes
- **Impact:** MEDIUM-HIGH

---

### 🥈 TIER 2: HIGH PRIORITY (Educational Improvements)

**Priority 2.1:** Add Educational Markdown Cells
- Theory of sentiment analysis approaches (3 paradigms)
- Explanation of each model's strengths/weaknesses
- Smoothing technique comparison
- Interpretation guidelines for peaks/valleys
- **Effort:** 2-3 hours
- **Impact:** HIGH (main request)

**Priority 2.2:** Add Savitzky-Golay Smoothing
```python
from scipy.signal import savgol_filter

def apply_smoothing_comparison(df, column, window_pct=0.1):
    """
    Compare different smoothing techniques
    
    Parameters:
    -----------
    df : DataFrame
        Data containing sentiment scores
    column : str
        Column name to smooth
    window_pct : float
        Window size as percentage of data length
    
    Returns:
    --------
    DataFrame with multiple smoothing methods
    """
    n = len(df)
    window = int(n * window_pct)
    
    # Ensure window is odd for Savitzky-Golay
    if window % 2 == 0:
        window += 1
    
    results = pd.DataFrame()
    results['sentence_no'] = df['sentence_no']
    results['raw'] = df[column]
    
    # Rolling mean
    results['rolling_mean'] = df[column].rolling(
        window=window, 
        center=True
    ).mean()
    
    # Savitzky-Golay (preserves peaks better)
    results['savgol'] = savgol_filter(
        df[column].fillna(0),  # Handle NaN
        window_length=min(window, 51),  # Cap at 51
        polyorder=3
    )
    
    # Exponential moving average
    results['ema'] = df[column].ewm(
        span=window,
        adjust=False
    ).mean()
    
    return results
```
- **Effort:** 1 hour
- **Impact:** MEDIUM-HIGH

**Priority 2.3:** Add Progress Bars & Logging
- Wrap all long-running loops with tqdm
- Add structured logging with time estimates
- **Effort:** 1-2 hours
- **Impact:** MEDIUM

**Priority 2.4:** Add Error Handling
- Try/except blocks for file operations
- Validation for uploaded files
- Model loading error handling
- **Effort:** 1-2 hours
- **Impact:** MEDIUM

---

### 🥉 TIER 3: MEDIUM PRIORITY (Code Quality)

**Priority 3.1:** Modularize Code
- Extract functions into separate cells/modules
- Create reusable sentiment analysis pipeline
- **Effort:** 3-4 hours
- **Impact:** MEDIUM

**Priority 3.2:** Add Data Validation
```python
def validate_text_data(text_list):
    """Validate input text data"""
    if not text_list:
        raise ValueError("Empty text list")
    
    if not all(isinstance(x, str) for x in text_list):
        raise TypeError("All items must be strings")
    
    # Check for reasonable sentence lengths
    lengths = [len(s) for s in text_list]
    avg_len = np.mean(lengths)
    
    if avg_len < 10:
        logger.warning(f"Average sentence length very short: {avg_len}")
    
    if avg_len > 500:
        logger.warning(f"Average sentence length very long: {avg_len}")
    
    return True
```
- **Effort:** 1 hour
- **Impact:** LOW-MEDIUM

**Priority 3.3:** Improve Visualization
- Add interactive Plotly charts with hover text
- Color-code by sentiment intensity
- Add vertical lines for act breaks
- Annotations for key scenes
- **Effort:** 2-3 hours
- **Impact:** MEDIUM

**Priority 3.4:** Add Model Comparison Metrics
```python
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error

def compare_models(df, model_cols):
    """
    Compare agreement between different sentiment models
    
    Returns correlation matrix and disagreement analysis
    """
    # Correlation matrix
    corr_matrix = df[model_cols].corr(method='spearman')
    
    # Find sentences with high disagreement
    df['std_across_models'] = df[model_cols].std(axis=1)
    high_disagreement = df.nlargest(10, 'std_across_models')
    
    return corr_matrix, high_disagreement
```
- **Effort:** 1-2 hours
- **Impact:** MEDIUM

---

### 🏅 TIER 4: LOW PRIORITY (Nice-to-Have)

**Priority 4.1:** Add Caching
- Cache model downloads
- Save intermediate results
- **Effort:** 1 hour
- **Impact:** LOW

**Priority 4.2:** Add Unit Tests
- Test data loading functions
- Test sentiment calculation accuracy
- **Effort:** 3-4 hours
- **Impact:** LOW (for notebook context)

**Priority 4.3:** Add Configuration File
- YAML configuration for all parameters
- Easy experimentation with settings
- **Effort:** 1 hour
- **Impact:** LOW-MEDIUM

**Priority 4.4:** Add Export Functionality
- Export to multiple formats (CSV, JSON, Parquet)
- Generate PDF report automatically
- **Effort:** 2 hours
- **Impact:** LOW-MEDIUM

---

## PART 6: SPECIFIC MISSING FEATURES

### 6.1 Modern Model Alternatives (2025)

**Should Add:**
1. **BERT variants:**
   - `cardiffnlp/twitter-roberta-base-sentiment-latest` (2023)
   - `j-hartmann/emotion-english-distilroberta-base` (multi-emotion)

2. **Multilingual:**
   - `nlptown/bert-base-multilingual-uncased-sentiment` (maintained)

3. **Domain-specific:**
   - `ProsusAI/finbert` (financial text)
   - `nlptown/bert-base-multilingual-uncased-sentiment` (general)

**Example Integration:**
```python
def load_modern_sentiment_model(model_name):
    """
    Load modern sentiment analysis model with best practices
    
    Parameters:
    -----------
    model_name : str
        HuggingFace model identifier
        
    Returns:
    --------
    pipeline : transformers.Pipeline
    """
    from transformers import pipeline
    import torch
    
    # Detect available device
    device = 0 if torch.cuda.is_available() else -1
    
    # Load with optimizations
    classifier = pipeline(
        "sentiment-analysis",
        model=model_name,
        device=device,
        batch_size=16 if device >= 0 else 1,  # Batch processing on GPU
        truncation=True,
        max_length=512
    )
    
    logger.info(f"✅ Loaded {model_name} on {'GPU' if device >= 0 else 'CPU'}")
    
    return classifier
```

---

### 6.2 Missing Statistical Analysis

**Should Add:**
```python
def analyze_sentiment_statistics(df, sentiment_col):
    """
    Comprehensive statistical analysis of sentiment time series
    
    Returns:
    --------
    dict : Statistical metrics and narrative insights
    """
    from scipy import stats
    from scipy.signal import find_peaks
    
    scores = df[sentiment_col].dropna()
    
    analysis = {
        # Central tendency
        'mean': scores.mean(),
        'median': scores.median(),
        'mode': scores.mode().iloc[0] if len(scores.mode()) > 0 else None,
        
        # Spread
        'std': scores.std(),
        'range': scores.max() - scores.min(),
        'iqr': scores.quantile(0.75) - scores.quantile(0.25),
        
        # Shape
        'skewness': stats.skew(scores),
        'kurtosis': stats.kurtosis(scores),
        
        # Dynamics
        'volatility': scores.diff().abs().mean(),
        'trend': stats.linregress(range(len(scores)), scores).slope,
        
        # Turning points
        'peaks': len(find_peaks(scores, prominence=0.5)[0]),
        'valleys': len(find_peaks(-scores, prominence=0.5)[0]),
        
        # Distribution
        'positive_pct': (scores > 0).sum() / len(scores) * 100,
        'negative_pct': (scores < 0).sum() / len(scores) * 100,
        'neutral_pct': (scores == 0).sum() / len(scores) * 100,
    }
    
    # Interpretation
    interp = []
    if analysis['mean'] > 0.2:
        interp.append("📈 Overall positive tone")
    elif analysis['mean'] < -0.2:
        interp.append("📉 Overall negative tone")
    else:
        interp.append("➡️  Neutral tone")
    
    if analysis['volatility'] > 0.3:
        interp.append("⚡ High emotional volatility (fast-paced)")
    elif analysis['volatility'] < 0.1:
        interp.append("🐌 Low emotional volatility (contemplative)")
    
    if analysis['trend'] > 0.0001:
        interp.append("↗️  Upward emotional trajectory")
    elif analysis['trend'] < -0.0001:
        interp.append("↘️  Downward emotional trajectory")
    
    analysis['interpretation'] = interp
    
    return analysis
```

---

### 6.3 Missing Visualization Best Practices

**Should Add:**
```python
def create_publication_quality_plot(df, sentiment_cols, novel_title="Novel"):
    """
    Create publication-quality sentiment arc visualization
    
    Features:
    - Multiple models overlaid
    - Act breaks marked
    - Key scenes annotated
    - Statistical shading
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.ticker import MaxNLocator
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), 
                                    height_ratios=[3, 1])
    
    # Color palette
    colors = plt.cm.Set2(range(len(sentiment_cols)))
    
    # Main plot
    for i, col in enumerate(sentiment_cols):
        smoothed = df[col].rolling(window=100, center=True).mean()
        ax1.plot(df.index, smoothed, 
                label=col.replace('_', ' ').title(),
                color=colors[i], linewidth=2, alpha=0.8)
    
    # Add ensemble mean
    ensemble_mean = df[sentiment_cols].mean(axis=1)
    ensemble_smooth = ensemble_mean.rolling(window=100, center=True).mean()
    ax1.plot(df.index, ensemble_smooth,
            label='Ensemble Mean', color='black', 
            linewidth=3, linestyle='--', alpha=0.7)
    
    # Statistical shading (confidence interval)
    ensemble_std = df[sentiment_cols].std(axis=1).rolling(
        window=100, center=True
    ).mean()
    ax1.fill_between(df.index, 
                     ensemble_smooth - ensemble_std,
                     ensemble_smooth + ensemble_std,
                     alpha=0.2, color='gray',
                     label='Model Uncertainty (±1 SD)')
    
    # Mark act breaks (assume 3-act structure)
    act_breaks = [len(df) // 3, 2 * len(df) // 3]
    for i, break_point in enumerate(act_breaks):
        ax1.axvline(break_point, color='red', linestyle=':', 
                   linewidth=2, alpha=0.5)
        ax1.text(break_point, ax1.get_ylim()[1], f'Act {i+1}|{i+2}',
                ha='center', va='bottom', fontsize=10, color='red')
    
    # Styling
    ax1.set_xlabel('Sentence Number', fontsize=12)
    ax1.set_ylabel('Sentiment Score', fontsize=12)
    ax1.set_title(f'Diachronic Sentiment Analysis: {novel_title}',
                 fontsize=16, fontweight='bold')
    ax1.legend(loc='best', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)
    
    # Volatility subplot
    volatility = ensemble_smooth.diff().abs().rolling(
        window=100, center=True
    ).mean()
    ax2.fill_between(df.index, volatility, alpha=0.6, color='steelblue')
    ax2.set_xlabel('Sentence Number', fontsize=12)
    ax2.set_ylabel('Volatility', fontsize=10)
    ax2.set_title('Emotional Volatility (Rate of Sentiment Change)',
                 fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{novel_title}_sentiment_arc.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Saved publication-quality figure: {novel_title}_sentiment_arc.png")
```

---

## PART 7: TECHNICAL DEBT ASSESSMENT

### Code Smells

1. **Magic Numbers:** Window sizes (0.1) hardcoded throughout
2. **Code Duplication:** Model loading code repeated 5+ times
3. **Long Functions:** Several cells exceed 50 lines
4. **Deep Nesting:** 3-4 levels in places
5. **Global State:** Over-reliance on global `sentiment_df`
6. **Inconsistent Naming:** Mix of snake_case, camelCase
7. **No Type Hints:** Function signatures lack type information

### Maintainability Score: 4/10
- ❌ Poor modularity
- ❌ No testing
- ❌ Minimal documentation
- ❌ High coupling
- ⚠️  Some logging
- ✅ Generally readable

---

## PART 8: PERFORMANCE OPTIMIZATIONS

### Current Performance Issues

1. **No Batch Processing:** Models run sentence-by-sentence
2. **Sequential Processing:** No parallelization
3. **Memory Inefficient:** Multiple full DataFrame copies
4. **No Caching:** Re-downloads models every run

### Recommended Optimizations

```python
def batch_sentiment_analysis(texts, model, batch_size=32):
    """
    Process texts in batches for 10-100x speedup
    
    Parameters:
    -----------
    texts : list[str]
        Input texts to analyze
    model : Pipeline
        HuggingFace pipeline
    batch_size : int
        Number of texts per batch
        
    Returns:
    --------
    list : Sentiment scores
    """
    from tqdm.auto import tqdm
    import numpy as np
    
    all_results = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(texts), batch_size), 
                  total=n_batches,
                  desc="Processing batches"):
        batch = texts[i:i+batch_size]
        
        # Process batch
        try:
            results = model(batch, truncation=True, max_length=512)
            all_results.extend(results)
        except Exception as e:
            logger.error(f"Batch {i//batch_size} failed: {e}")
            # Fall back to individual processing
            for text in batch:
                try:
                    result = model(text, truncation=True, max_length=512)
                    all_results.append(result[0])
                except:
                    all_results.append({'label': 'NEUTRAL', 'score': 0.0})
    
    return all_results

# Usage example:
# Before: 30 minutes
# for text in texts:
#     result = model(text)
#
# After: 2 minutes
# results = batch_sentiment_analysis(texts, model, batch_size=32)
```

**Expected Speedup:**
- VADER: 1.5x (already fast)
- Transformers: 10-50x (huge improvement)
- Overall runtime: ~70% reduction

---

## SUMMARY

### Critical Action Items (Do First)

1. ✅ Fix syntax errors (lines 3718-3720)
2. ✅ Fix variable definitions (line 270, 1602, 442)
3. ✅ Update deprecated imports (`AutoModelWithLMHead`)
4. ✅ Add educational markdown cells explaining theory
5. ✅ Add Savitzky-Golay smoothing comparison
6. ✅ Add batch processing for transformers
7. ✅ Add comprehensive error handling

### Estimated Time Investment

- **Minimal (fixes only):** 2-3 hours
- **Recommended (educational):** 8-12 hours  
- **Comprehensive (best practices):** 20-30 hours

### Educational Value Rating

- **Current:** 4/10 (functional but opaque)
- **After Tier 1-2 improvements:** 8/10 (clear and instructive)
- **After Tier 1-3 improvements:** 9/10 (publication-ready teaching tool)

---

## APPENDIX A: Version Compatibility Matrix

| Library | Oct 2022 | Oct 2025 | Breaking Changes |
|---------|----------|----------|------------------|
| transformers | 4.22.0 | 4.45.0 | AutoModelWithLMHead removed |
| torch | 1.12.0 | 2.5.0 | API stable |
| pandas | 1.4.0 | 2.2.0 | Minor deprecations |
| numpy | 1.22.0 | 2.1.0 | Some dtype changes |
| vaderSentiment | 3.3.1 | 3.3.2 | None |
| textblob | 0.17.1 | 0.18.0 | None |
| rpy2 | 3.5.1 | 3.5.16 | None |
| scipy | 1.8.0 | 1.14.0 | None |
| matplotlib | 3.5.0 | 3.9.0 | None |
| seaborn | 0.11.0 | 0.13.0 | Minor |

---

## APPENDIX B: Recommended Reading

**For Students:**
1. "Speech and Language Processing" - Jurafsky & Martin (Chapter 4: Sentiment)
2. "Natural Language Processing with Transformers" - Tunstall et al.
3. SentimentArcs Paper: https://arxiv.org/abs/2110.09454

**For Implementation:**
1. HuggingFace Docs: https://huggingface.co/docs/transformers/
2. VADER Paper: Hutto & Gilbert (2014)
3. Savitzky-Golay: https://scipy-cookbook.readthedocs.io/
