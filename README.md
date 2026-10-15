# Trolls, Haters, or Teammates — Sentiment Classification of PvP Game Chat

ML mini project that re-implements *"Trolls, Haters, or Teammates: Classifying Sentiment in the Language of PvP Game Communications"* on League of Legends and DOTA 2 in-game chat.

Each chat message is classified into one of three classes:

| Label | Class | Examples |
|---|---|---|
| 0 | Neutral / positive | `gg wp`, `ty`, `nice gank` |
| 1 | Negative attitude | `noob team`, `report jax`, `stop feeding` |
| 2 | Derogatory | slurs, sexist / racist / homophobic abuse |

## Repository structure
```
PvP_Toxicity_Classification.ipynb   main notebook: preprocessing → models → evaluation
prepare_dataset.py                  builds the labelled dataset from the raw Kaggle CSVs
data/pvp_chat_labelled.csv          33,000 labelled messages (output of prepare_dataset.py)
outputs/                            figures + results table produced by the notebook (added in Part 2)
requirements.txt
```

## Datasets
- League of Legends Tribunal chat logs, Kaggle (`chatlogs.csv`, 1.7M messages)
- DOTA 2 match chat, Kaggle (`dota2_chat_messages.csv`, 21M messages)

The raw files have **no per-message sentiment labels**. `prepare_dataset.py`:
1. Keeps Latin-script messages that contain at least one word.
2. Removes duplicate messages, so the same message cannot appear in both train and test.
3. **Weak-labels** each message with a gaming-toxicity lexicon, similar to the paper's hand-built dictionary.
4. Samples 33,000 messages at the paper's class balance (44% / 31% / 25%).

## How to run
**Google Colab (recommended)**
1. Open `PvP_Toxicity_Classification.ipynb` in Colab (File → Upload notebook).
2. Click *Runtime → Run all*. When asked, upload `data/pvp_chat_labelled.csv`.

**Local**
```bash
pip install -r requirements.txt
# optional: rebuild the dataset from the raw Kaggle files
python prepare_dataset.py --raw_dir path/to/raw_csvs --out data/pvp_chat_labelled.csv
jupyter notebook PvP_Toxicity_Classification.ipynb
```

## Method
- **Pre-processing:** lowercasing, URL removal, squeezing repeated letters (`noooob` → `noob`), stratified 60/20/20 train/validation/test split.
- **Features:** TF-IDF on word unigrams and bigrams (5,000 features).
- **Models (as in the paper):**
  - Binary SVM baseline (linear, RBF and polynomial kernels)
  - Naive Bayes (fixed dictionary, and TF-IDF)
  - KNN (K = 1, 2, 3, 6, Euclidean distance)
  - Random Forest (Gini, entropy, and entropy with max depth 32)
- **Model selection:** chosen on validation accuracy, reported on the held-out test set.
- **Our addition: improving KNN.** TruncatedSVD dimensionality reduction plus a grid search over K and the distance weighting.

## Results (test set, 6,600 messages)
| Model | Accuracy |
|---|---|
| Binary SVM, linear (toxic vs not) | 0.986 |
| Naive Bayes (dictionary) | 0.923 |
| Naive Bayes (TF-IDF) | 0.905 |
| KNN (K = 6, original) | 0.660 |
| **KNN improved (SVD 100 dims, K = 7, distance-weighted)** | **0.815** |
| Random Forest (Entropy, depth 32) | 0.853 |
| Random Forest (Entropy) | 0.963 |
| **Random Forest (Gini)**, the best model | **0.967** |

**Key findings**
- Random Forest is the strongest model, as in the paper. Gini and entropy are nearly identical, while capping the depth hurts.
- KNN is weak on 5,000-dimensional sparse TF-IDF vectors. Reducing them to 100 dimensions with SVD makes distances meaningful and raises accuracy by about 15 points.
- KNN does worse than in the paper because we removed duplicate messages before splitting. The paper's split likely contained exact repeats, which K = 1 can simply look up.

## Limitations
- The labels come from a lexicon, not human annotators.
- The lexicon is English-only, and some non-English Latin-script chat remains in the data.
- Sarcasm and friendly banter are hard to classify.

## Individual Contributions
**Aadarsh Reddy Yedhala (PES1UG24CS706): data pipeline and baseline models**
- Collected and merged the League of Legends and DOTA 2 chat datasets from Kaggle
- Wrote the dataset preparation script: language filtering, removing duplicates, labelling with a toxicity word list, and class-balanced sampling (`prepare_dataset.py`)
- Built the text pre-processing (lowercasing, URL removal, shortening repeated letters) and the stratified 60/20/20 train/validation/test split
- Implemented the TF-IDF feature extraction (single words and word pairs)
- Implemented the binary SVM baseline with linear, RBF and polynomial kernels
- Implemented Naive Bayes, once on a fixed dictionary and once on TF-IDF, plus the most telling words for each class
- Set up the GitHub repository and wrote the README

**Ganavi Gowda (PES1UG24CS707) : model evaluation and improvement**
- Implemented K-Nearest Neighbours (K = 1, 2, 3, 6) with Euclidean distance
- Implemented Random Forest with Gini, entropy, and depth-limited settings
- Built the model comparison: accuracy and per-class F1 table, F1 chart, confusion matrices, and best-model selection on validation accuracy
- Performed error analysis on misclassified messages
- Improved KNN accuracy from 0.66 to 0.815 using dimensionality reduction (TruncatedSVD) and grid search over K and the distance weighting

**Joint work**
- Literature review of the reference paper
- Interpretation of results and the project write-up
