# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a sentiment analysis research project focused on **diachronic sentiment arcs** - tracking how sentiment changes over time in texts. The project contains Jupyter notebooks that implement educational materials for teaching text mining and sentiment analysis to non-STEM students.

## Architecture

### Core Components

1. **Sentiment Analysis Methods**:
   - **VADER**: Lexicon-based sentiment analysis (fast, transparent)
   - **Classic ML**: TF-IDF + Logistic Regression baseline (scaffold provided)
   - **Transformers**: DistilBERT fine-tuned on SST-2 (contextual, higher accuracy)

2. **Time Series Processing**:
   - Sentence segmentation and scoring
   - Multiple smoothing methods: Moving Average, Exponential Moving Average, Savitzky-Golay filtering
   - Peak/valley detection with minimum spacing constraints
   - Bootstrap uncertainty bands for robustness analysis

3. **Analysis Pipeline**:
   - Text cleaning and sentence segmentation (`pysbd` with fallback)
   - Normalization using z-scores
   - Model agreement analysis using Spearman correlation
   - Interactive and static visualization

### Key Files

- `Diachronic_Sentiment_Arc_2025_gpt5_20251028.ipynb`: Main teaching notebook with complete implementation
- `sentimentarcs_update2025_gpt5_BASE_20251028.ipynb`: Base version (identical content)

## Running the Notebooks

### Environment Setup

The notebooks include automated dependency installation. Key requirements are installed via:

```bash
pip install "transformers>=4.44,<5" "datasets>=3.0,<3.1" "accelerate>=1.0" \
                "vaderSentiment>=3.3.2" "textblob>=0.18" "matplotlib>=3.9" \
                "plotly>=5.24" "pysbd>=0.3" "scipy>=1.13" "tqdm>=4.66" \
                "scikit-learn>=1.5"
```

Plus NLTK data downloads for `punkt` and `vader_lexicon`.

### Configuration

All parameters are centralized in a `Config` dataclass:
- `model_name`: `"distilbert-base-uncased-finetuned-sst-2-english"` (binary sentiment classifier)
- `smoothing`: `"savgol"` (options: `"ma"`, `"ema"`, `"savgol"`)
- `window_frac`: `0.10` (smoothing window as fraction of series length)
- `batch_size`: `32` for transformer inference
- `interactive`: `True` enables Plotly visualizations

### Running Individual Components

1. **Basic sentiment analysis**: Run cells through section 4.1 for VADER-only analysis
2. **Full pipeline**: Execute through section 7 for peaks + uncertainty analysis
3. **Model comparison**: Section 8 provides correlation analysis between methods

## Key Implementation Details

### Smoothing Algorithm
The project uses **Savitzky-Golay filtering** as the default smoothing method, which preserves peaks better than simple moving averages while reducing noise. Window size is automatically calculated as 10% of series length (forced to be odd).

### Bootstrap Uncertainty
Uses 200 bootstrap resamples to create uncertainty bands (±1 SD) around the smoothed sentiment curve, providing robustness checks against over-interpretation.

### Model Agreement
Analyzes correlation between VADER and transformer scores using Spearman correlation, helping identify texts where different methods disagree.

## Pedagogical Design

The notebooks are designed for teaching non-technical students:
- Each step has clear explanations of assumptions and limitations
- Includes ethics section discussing domain shift, cultural bias, and over-interpretation risks
- Provides both quick-start paths and comprehensive analysis options
- Emphasizes that sentiment arcs are descriptive, not causal

## Dependencies

- **Core**: `transformers`, `torch`, `pandas`, `numpy`, `scipy`
- **Sentiment**: `vaderSentiment`, `textblob`
- **Visualization**: `matplotlib`, `plotly`
- **Text processing**: `pysbd`, `nltk`
- **ML**: `scikit-learn` (for baseline scaffold)

## Output Structure

Results are saved to `./outputs/` directory (created automatically). The notebooks generate:
- Processed DataFrames with sentence-level scores
- Visualizations (static matplotlib and interactive plotly)
- Peak/valley annotations
- Bootstrap uncertainty bands