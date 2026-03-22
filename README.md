# Identifying B2B Purchase Intent from Software Reviews Using NLP
 
> MSc Data Science, AI & Digital Business — GISMA University of Applied Sciences   

 
## What This Project Is About
 
B2B software buyers leave a detailed written record of their purchasing journey every time they write a review on G2.com. This thesis tests whether NLP can automatically detect what stage of the buying process the reviewer is in, are they about to buy? About to leave? Actively comparing tools?, using only the words they already wrote.
 
The project builds a labeled dataset from scratch, trains and compares 11 experiments across 6 methodological families, and demonstrates that a fine-tuned transformer model (DistilBERT) can detect B2B purchase intent with **94% accuracy**.

  
## Intent Categories
 
Five purchase intent categories were defined and labeled:
 
| Category | Description | Example signal |
|---|---|---|
| High Purchase Intent | Reviewer chose, purchased, or switched TO this product | *"after evaluating several options, we went with..."* |
| Churn Intent | Reviewer is leaving or looking for alternatives | *"we are actively looking for alternatives..."* |
| Evaluation Intent | Reviewer is trialing or comparing | *"during our free trial we noticed..."* |
| Advocacy Intent | Reviewer is recommending the product to others | *"I highly recommend this to any team..."* |
| Neutral | General experience, no clear intent signal | *"the interface is clean and easy to use..."* |

  
## Dataset
 
### Phase 1: Original (7 products)
- **Scraper:** Apify G2 Product Scraper (capped at 1,000/product)
- **Raw reviews:** 7,000
- **Products:** HubSpot, Salesforce, Pipedrive, Intercom, ActiveCampaign, Zendesk, Mixpanel
- **Balanced dataset:** 1,758 reviews
 
### Phase 2: Expanded (15 products)
- **Scraper:** [Apify G2 Explorer](https://apify.com/jupri/g2-explorer) no per-product cap
- **Raw reviews:** 103,439
- **Products:** HubSpot Marketing Hub · Salesforce Sales Cloud · ActiveCampaign · Slack · Zoom Workplace · Microsoft Teams · Monday.com · Notion · Asana · Pipedrive · Shopify · Confluence · Mixpanel · Intercom · Zendesk
- **Balanced dataset:** 12,154 reviews
- b2b_reviews_expanded_balanced.csv Too large for GitHub, available on Google Drive: https://drive.google.com/file/d/1l8TvD1sotu15Yt21UiZUS-IOgs7sOA0g/view?usp=sharing
 
### Dataset Journey
 
```
103,439 raw to 5,115 labeled on raw text (5%) to 15,069 after cleaning (14.6%) to 12,154 after balancing
```
 
**Key finding:** Text cleaning nearly tripled labeling yield from 5% to 14.6%. Stripping G2 question headers allowed keyword matching to reach the reviewer's actual words.
 
 
## Results Summary
 
### Supervised Models
 
| Model | Old Dataset (1,758) | New Dataset (12,154) |
|---|---|---|
| Logistic Regression | 57% / F1: 0.52 | 72% / F1: 0.70 |
| SVM | 62% / F1: 0.60 | 83% / F1: 0.83 |
| **DistilBERT** ⭐ | 71% / F1: 0.67 | **94% / F1: 0.94** |
 
### Low-Resource Approaches
 
| Experiment | Old Dataset | New Dataset |
|---|---|---|
| Zero-shot (BART-MNLI) | 41% | 27% |
| Few-shot k=3 | 20% | 25% |
| Few-shot k=5 | 23% | 27% |
| Few-shot k=10 | 22% | 28% |
| Synthetic data | 44% | — |
| LLM Prompting (Gemini) | — | 20%* |
 
*Rate limited on free tier, all predictions defaulted to Neutral
 
### Additional Experiments
 
| Experiment | Finding |
|---|---|
| Raw vs Cleaned | Unbalanced LR: 97% accuracy but only 5% Churn recall, confirms balancing is essential |
| Error Analysis | 154 DistilBERT errors, 52% are intent predicted as Neutral (boundary ambiguity, not model failure) |
| Heuristic Variance | SVM: 50% at n=100 → 83% at n=9,723. Monotonic improvement, no plateau |
 
 
## Key Findings
 
1. **Fine-tuned transformers work even with automated labels**: DistilBERT hits 94% using keyword-based weak supervision, no human annotation needed
2. **More data beats a better model**: SVM at 9,700 reviews (83%) outperforms DistilBERT at 1,758 reviews (71%)
3. **The few-shot valley is real**: k=3 drops to 20%, worse than zero-shot (41%). Don't use few-shot with fewer than ~50 examples per class
4. **Intent is a continuum**: 52% of errors are at Advocacy/Neutral and Evaluation/Neutral boundaries, reflecting genuine semantic overlap
 
  
 
## How to Run
 
All experiments were run on **Google Colab with a T4 GPU** (free tier).
 
1. Open the notebook in Google Colab
2. Mount your Google Drive when prompted
3. Upload the dataset CSV to `/content/drive/MyDrive/Datasets/`
4. Run cells in order — each experiment is clearly labeled
 
**Requirements** (all pre-installed on Colab):
```
transformers
torch
scikit-learn
pandas
numpy
sentence-transformers
bertopic
matplotlib
seaborn
```
 
 
## Scraping Setup (Apify)
 
To replicate the data collection:
 
1. Create a free account at [apify.com](https://apify.com)
2. Open the [G2 Explorer actor](https://apify.com/jupri/g2-explorer)
3. Set memory to **1,024MB** (default 128MB causes crashes)
4. Enable **No timeout**
5. Disable the **switched filter**
6. Add G2 product URLs and set limit to 10,000 per product
 
 
## Model Details
 
**DistilBERT fine-tuning config:**
- Model: `distilbert-base-uncased`
- Epochs: 3
- Batch size: 16
- Learning rate: 2e-5
- Max token length: 256
- Optimiser: AdamW
- Hardware: Google Colab T4 GPU

  
## References
 
Key papers used in this thesis:
 
- Devlin et al. (2019): BERT
- Sanh et al. (2019): DistilBERT
- Brown et al. (2020): Few-shot learners
- Blei et al. (2003): LDA
- Grootendorst (2022): BERTopic
- Ratner et al. (2016): Weak supervision / data programming
- Casanueva et al. (2020): Intent detection with dual encoders
 
Full reference list available in the thesis document (Chapter 6).
