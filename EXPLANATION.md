# 📰 NewsAnalyzer — Complete Code Explanation

> This document explains every cell of the `news_analyzer.ipynb` notebook in detail.
> Written as a personal learning reference — covers what each line does, why we wrote it,
> and the reasoning behind every decision.

---

## 🗺️ What Are We Building and Why?

Before diving into code — let's understand the problem.

Imagine you have **18,846 raw news articles** dumped in front of you with no organization:

```
Article 1: "NASA launched shuttle..."
Article 2: "Hockey team scored..."
Article 3: "God and religion..."
... 18,843 more articles
```

Three real world questions arise:

**Question 1 — "What topics exist in this data?"**
We don't know what's inside! NMF answers this automatically — no labels needed. It discovers topics like space, sports, religion, and crypto on its own.

**Question 2 — "Can I visualize how similar articles cluster?"**
Each article has 5000 numbers (TF-IDF features) — impossible to plot directly! We use TSVD to reduce to 100, then t-SNE to reduce to 2 — now we can see clusters visually!

**Question 3 — "Given a new article — which category is it?"**
A new article arrives → TSVD + LDA/SVM predicts → "This is a SPORTS article!" 🏒

**Real World Connection:**

| Company | What They Do | What We Built |
|---|---|---|
| Google News | Groups similar news | Our t-SNE clusters |
| Twitter | Finds trending topics | Our NMF topics |
| Gmail | Categorizes emails | Our LDA classifier |

---

## 📦 Cell 1 — Imports

```python
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
```

### Why each import:

**Group 1 — Data:**
- `fetch_20newsgroups` → loads 18,846 news articles directly from sklearn — no manual download needed
- `pandas` → for DataFrame (our data table)
- `numpy` → for array operations and math

**Group 2 — Visualization:**
- `matplotlib.pyplot` → basic plotting (scatter plots, bar charts)
- `seaborn` → beautiful statistical plots (our category countplot!)

**Group 3 — Text Processing:**
- `re` → regex library for cleaning text (remove emails, URLs, special characters)
- `TfidfVectorizer` → converts raw text → numbers (our 18846×5000 matrix)

**Group 4 — The 5 Dimensionality Reduction Techniques:**
- `TruncatedSVD` → TSVD (from `sklearn.decomposition`)
- `NMF` → Non-Negative Matrix Factorization (from `sklearn.decomposition`)
- `PCA` → Principal Component Analysis (from `sklearn.decomposition`)
- `TSNE` → t-SNE visualization (from `sklearn.manifold` — manifold learning!)
- `LinearDiscriminantAnalysis` → LDA (from `sklearn.discriminant_analysis`)

> **Important:** PCA, TSVD, NMF all come from `sklearn.decomposition`. LDA comes from `sklearn.discriminant_analysis` because it's also a classifier! t-SNE comes from `sklearn.manifold` because it's non-linear.

**Group 5 — ML Training & Evaluation:**
- `train_test_split` → split data 80% train / 20% test
- `classification_report` → precision, recall, F1 per category
- `accuracy_score` → overall accuracy percentage
- `LinearSVC` → our SVM classifier
- `Pipeline` → chain multiple steps together

---

## 📥 Cell 2 — Loading the Dataset

```python
data = fetch_20newsgroups(subset='all')
```

### What happens:
- **First time** → downloads ~14MB dataset from internet → saves to local cache
- **Every time after** → loads from cache instantly (no re-download!) ⚡

### The `subset` parameter:

| Parameter | Articles Loaded |
|---|---|
| `subset='train'` | ~11,314 (training portion only) |
| `subset='test'` | ~7,532 (test portion only) |
| `subset='all'` | ~18,846 (everything!) ✅ |

We use `'all'` because we want maximum data. We'll do our own train/test split later.

### What `data` object contains:
```python
data.data         → list of 18,846 raw article texts
data.target       → array of category numbers [0, 1, 2, ... 19]
data.target_names → list of 20 category names
data.filenames    → file paths of each article
```

---

## 🗂️ Cell 3 & 5 — Building the DataFrame

### Cell 3 (Wrong approach — first attempt):
```python
df = pd.DataFrame(data.data)
```

This creates a DataFrame with only **1 column** — just the raw text! We lost the labels entirely. A DataFrame with only text is useless for ML.

### Cell 5 (Correct approach):
```python
df = pd.DataFrame({
    'text': data.data,
    'label': data.target,
    'category': [data.target_names[i] for i in data.target]
})
```

Now we have 3 proper columns — text, label number, and category name.

### Understanding `data.target` vs `data.target_names`:

These are two completely separate things:

```python
data.target = [0, 11, 3, 15, 7, ...]
# length = 18,846 (one number per article)
# tells us WHICH category each article belongs to

data.target_names = ['alt.atheism', 'comp.graphics', ...]
# length = 20 (one name per category)
# tells us WHAT each category number means
```

Think of it like a dictionary:
```
Index 0  → 'alt.atheism'
Index 1  → 'comp.graphics'
...
Index 19 → 'talk.religion.misc'
```

### The list comprehension explained:
```python
[data.target_names[i] for i in data.target]

# data.target = [0, 11, 3, ...]
# data.target_names[0]  = 'alt.atheism'
# data.target_names[11] = 'rec.sport.hockey'
# data.target_names[3]  = 'comp.sys.mac.hardware'

# Maps each article's number → its actual category name!
```

---

## 📊 Cell 9 — Category Distribution Plot

```python
plt.figure(figsize=(16, 6))
sns.countplot(x='category', data=df, color='teal')
plt.title("Distribution of News Categories", fontsize=14)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
```

### Line by line:

**`plt.figure(figsize=(16, 6))`**
Creates blank canvas BEFORE plotting. Must come first! 16 wide × 6 tall — wide enough for 20 category names.

**`sns.countplot(x='category', data=df, color='teal')`**
Automatically counts occurrences of each category and draws a bar. Internally does `df['category'].value_counts()` — no manual counting needed!

**`plt.xticks(rotation=90)`**
Rotates x-axis labels 90 degrees. Without this — labels overlap and become unreadable!

**`plt.tight_layout()`**
Adjusts spacing automatically so rotated labels don't get cut off at the bottom.

### What we found:
- Most categories have ~1000 articles → fairly balanced dataset ✅
- `talk.religion.misc` has only ~630 — smallest category
- Balanced dataset = no class imbalance problems for ML!

---

## 🧹 Cell 10 & 11 — Text Cleaning

```python
def clean_text(text):
    text = re.sub(r'\S+@\S+', '', text)       # Remove emails
    text = re.sub(r'http\S+', '', text)        # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)   # Remove special chars
    text = re.sub(r'\s+', ' ', text).strip()  # Fix whitespace
    text = text.lower()                        # Lowercase
    return text

df["cleaned_text"] = df["text"].apply(clean_text)
```

### How `re.sub()` works:
```python
re.sub(pattern, replacement, string)
# Finds pattern → replaces with replacement → returns clean string
```

### Each cleaning step:

**Step 1 — Remove emails:**
```python
re.sub(r'\S+@\S+', '', text)
# \S+ = one or more non-whitespace chars
# @   = literal @ symbol
# Matches: user@email.com, mr47+@andrew.cmu.edu
# Before: "From: user@email.com wrote:"
# After:  "From:  wrote:"
```

**Step 2 — Remove URLs:**
```python
re.sub(r'http\S+', '', text)
# Matches: http://www.google.com, https://github.com
# Before: "visit http://www.nasa.gov for more"
# After:  "visit  for more"
```

**Step 3 — Remove special characters and numbers:**
```python
re.sub(r'[^a-zA-Z\s]', '', text)
# [^...] = match anything NOT in this set
# Removes: numbers, punctuation, symbols
# Before: "NASA launched 3 rockets! Cost: $4.5 billion"
# After:  "NASA launched  rockets Cost  billion"
```

**Step 4 — Fix whitespace:**
```python
re.sub(r'\s+', ' ', text).strip()
# \s+ = one or more whitespace chars
# Replaces multiple spaces with single space
# .strip() removes leading/trailing spaces
```

**Step 5 — Lowercase:**
```python
text.lower()
# "NASA" and "nasa" and "Nasa" → all become "nasa"
# Treated as same word by TF-IDF! ✅
```

### Common bug to avoid:
```python
# WRONG — trailing commas make tuples!
text = re.sub(r'\S+@\S+', '', text),   # ← comma! tuple!

# CORRECT
text = re.sub(r'\S+@\S+', '', text)    # ← no comma ✅
```

### The `.apply()` call:
```python
df["cleaned_text"] = df["text"].apply(clean_text)

# .apply() runs clean_text on every row automatically
# No manual loop needed!
# pandas handles all 18,846 iterations internally ✅
```

---

## 🔢 Cell 15 & 17 — TF-IDF Vectorization

```python
tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    min_df=5,
    max_df=0.95
)

X = tfidf.fit_transform(df['cleaned_text'])
y = df['label'].values
```

### The 4 parameters:

**`max_features=5000`**
Keep only top 5000 most important words. Many rare words = noise. 5000 is the sweet spot for this dataset.

**`stop_words='english'`**
Removes ~318 common English words automatically: "the", "is", "are", "a", "an", "in", "on"...

**`min_df=5`**
Ignore words appearing in fewer than 5 articles. Too rare = probably typos or very specific names. Not useful for general classification.

**`max_df=0.95`**
Ignore words appearing in more than 95% of articles. If a word is everywhere — it can't distinguish categories!

### `fit_transform()` does two things:
```python
# fit()      → learns vocabulary (which 5000 words to keep)
#              calculates IDF for each word
# transform() → converts articles to numbers
#              creates 18846 × 5000 matrix

# fit_transform() = both in one step! ✅
```

### Critical rule for test data:
```python
# Training data → fit_transform() (learn + convert)
tfidf.fit_transform(X_train)

# Test data → transform() ONLY! (convert only!)
tfidf.transform(X_test)

# Never fit on test data → data leakage! ❌
```

### The result:
```
X = sparse matrix (18846 × 5000)
    rows    = articles
    columns = 5000 most important words
    values  = TF-IDF weights (0.0 to 1.0)

y = numpy array (18846,)
    values  = category numbers (0 to 19)
```

### TF-IDF formula:
```
TF  = how often word appears in THIS article
IDF = log(total articles / articles containing this word)
TF-IDF = TF × IDF

High TF-IDF → frequent in THIS article, rare overall = important!
Low TF-IDF  → common everywhere = not useful for distinguishing!
```

### Why sparse matrix?
```
18846 articles × 5000 words = 94,230,000 cells
BUT each article uses only ~200-300 unique words
So 99% of cells = 0!

Sparse matrix stores only non-zero values
→ saves massive amounts of memory! ✅
```

---

## 📉 Cell 21 — TSVD (Truncated SVD)

```python
tsvd = TruncatedSVD(n_components=300, random_state=42)
X_tsvd = tsvd.fit_transform(X)

print("Original shape:", X.shape)
print("After TSVD shape:", X_tsvd.shape)
print("Total variance retained:",
      round(tsvd.explained_variance_ratio_.sum() * 100, 2), "%")
```

### Parameters:

**`n_components=300`**
Reduce from 5000 features → 300 components. First tried 100 → only 15.56% variance. Then tried 300 → 30.88% variance (better balance of speed vs information).

**`random_state=42`**
TSVD uses randomized algorithms internally. Fixed seed = same results every run = reproducible! ✅

### Why TSVD and NOT PCA for text:
```python
# X is SPARSE matrix!

# PCA on sparse:
pca.fit_transform(X)            # crashes or uses huge RAM!
pca.fit_transform(X.toarray())  # 754MB dense matrix! 💥

# TSVD on sparse:
tsvd.fit_transform(X)           # works perfectly! ✅
```

### What TSVD does internally:
```
X = U × Σ × V^T  (full SVD — expensive!)

Truncated:
Keep only top 300 singular values from Σ
Throw away 4700 less important ones!

Result: (18846 × 300) dense matrix ✅
```

### The variance line explained:
```python
round(tsvd.explained_variance_ratio_.sum() * 100, 2)

# tsvd.explained_variance_ratio_ = array of 300 values
# Each = how much variance that component captures
# .sum() = add all 300 values together = 0.3088
# * 100  = convert to percentage = 30.88
# round(,2) = 30.88% ✅
```

### Why only 30.88% for text?
```
Numeric data → features correlated → variance concentrated
100 components → 95% variance ✅

Text data → each word mostly independent
Variance spread across thousands of dimensions
300 components → 30.88% variance

This is NORMAL and EXPECTED for text!
30.88% still captures the most important patterns! ✅
```

---

## 🎯 Cell 22, 23, 24 — NMF (Topic Discovery)

```python
# Cell 22
nmf = NMF(n_components=20, random_state=42, max_iter=400)
X_nmf = nmf.fit_transform(X)

# Cell 23
feature_names = tfidf.get_feature_names_out()

# Cell 24
print("=== DISCOVERED TOPICS ===\n")
for topic_idx, topic in enumerate(nmf.components_):
    top_words = [feature_names[i]
                 for i in topic.argsort()[-10:][::-1]]
    print(f"Topic {topic_idx + 1:2d}: {' | '.join(top_words)}")
```

### NMF Parameters:

**`n_components=20`**
Discover exactly 20 topics. We chose 20 because dataset has 20 categories — hoping each topic matches one category!

**`max_iter=400`**
NMF learns by iteratively improving its matrices. Default 200 sometimes not enough. 400 ensures full convergence.

### What NMF produces:
```
X ≈ W × H

W = X_nmf (18846 × 20) — Document-Topic matrix
"How much of each topic is in each article"

H = nmf.components_ (20 × 5000) — Topic-Word matrix
"How important each word is for each topic"
```

### Visual example:
```
W (Document-Topic):
           Topic1  Topic2  Topic20
Article1:   0.8     0.0  ...  0.0   ← mostly topic 1 (space)
Article2:   0.0     0.7  ...  0.0   ← mostly topic 2 (religion)

H (Topic-Word):
         "space" "god" "hockey"
Topic1:   0.9    0.0    0.0    ← space topic
Topic2:   0.0    0.8    0.0    ← religion topic
Topic3:   0.0    0.0    0.9    ← hockey topic
```

### Breaking down the topic printing loop:

**`feature_names = tfidf.get_feature_names_out()`**
Returns all 5000 words TF-IDF learned — the column names of our matrix!

**`enumerate(nmf.components_)`**
Gives both index AND value:
- `topic_idx` = 0, 1, 2... 19 (topic number)
- `topic` = array of 5000 importance values

**`topic.argsort()`**
Returns indices that sort array from smallest to largest.

**`[-10:]`**
Gets last 10 indices = indices of TOP 10 most important words (since argsort goes small→large, last = biggest).

**`[::-1]`**
Reverses order so most important word comes FIRST.

**`[feature_names[i] for i in ...]`**
Converts indices → actual words!

**`f"Topic {topic_idx + 1:2d}: ..."`**
- `+1` → start from 1 not 0
- `:2d` → always 2 digits wide so topics align nicely
- `' | '.join(top_words)` → joins words with " | " separator

### What NMF discovered (our results):
```
Topic  1: god | jesus | bible | christian     → Religion ✅
Topic  3: game | team | hockey | baseball      → Sports ✅
Topic  4: key | clipper | encryption | crypto  → Cryptography ✅
Topic 10: car | engine | dealer | ford         → Automobiles ✅
Topic 15: space | shuttle | nasa | moon        → Space ✅
Topic 20: bike | motorcycle | bmw | ride       → Motorcycles ✅
```

19 out of 20 topics were perfectly meaningful — without any labels! 🤩

---

## 🎨 Cell 25 — Selecting Subset for Visualization

```python
selected_categories = [
    'sci.space', 'rec.sport.hockey',
    'talk.religion.misc', 'comp.graphics',
    'rec.autos', 'sci.med'
]

mask = df['category'].isin(selected_categories)
df_subset = df[mask].reset_index(drop=True)
X_subset = X[mask.values]
y_subset = df_subset['label'].values
```

### Why only 6 categories?
```
20 categories = 20 colors = confusing messy plot!
6 categories  = 6 colors  = clean readable plot ✅

Chose 6 most DIFFERENT categories:
space, hockey, religion, computers, cars, medicine
Each uses completely different vocabulary
= should form distinct clusters!
```

### Key operations:

**`.isin(selected_categories)`**
Returns True/False for each row — True if category is in our list.

**`df[mask]`**
Boolean indexing — keeps only rows where mask = True.

**`reset_index(drop=True)`**
After filtering, index has gaps (0, 5, 11, 23...). Reset makes it sequential (0, 1, 2, 3...). `drop=True` prevents old index becoming a column.

**`X[mask.values]`**
Filter TF-IDF matrix same way as DataFrame. `.values` converts pandas Series → numpy array (needed for sparse matrix indexing).

---

## 📊 Cell 26 — PCA Visualization (Failed!)

```python
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_subset.toarray())
```

### Why `.toarray()`?
```python
# X_subset is SPARSE matrix
# PCA cannot work on sparse! Must convert to dense first

# For our subset:
# 5567 × 5000 × 8 bytes = ~222MB (manageable for subset)

# For full dataset:
# 18846 × 5000 × 8 bytes = ~754MB 💥 (too much!)
# This is why we only use subset for PCA!
```

### Why PCA failed:
```
Variance Component 1: 0.87%
Variance Component 2: 0.71%
Total:                1.58%

98.42% information LOST!
Clusters cannot form → everything blobs together ❌

Reason: Text data variance is spread across
thousands of dimensions very thinly.
PCA can only find 2 of those dimensions.
```

### The plotting loop:
```python
for i, category in enumerate(selected_categories):
    mask_cat = df_subset['category'] == category
    plt.scatter(
        X_pca[mask_cat, 0],  # x = Component 1
        X_pca[mask_cat, 1],  # y = Component 2
        c=colors[i],         # one color per category
        alpha=0.5,           # 50% transparent (overlapping dots)
        s=10                 # small dots (5567 points!)
    )
```

**`X_pca[mask_cat, 0]`**
Gets x-coordinates (Component 1) for ONE category only. `mask_cat` filters rows, `0` selects first column.

**`alpha=0.5`**
Transparency! Without this — overlapping dots form a solid blob. With transparency you can see density patterns.

---

## 🔧 Cell 27 — TSVD → PCA (Still Failed!)

```python
tsvd_viz = TruncatedSVD(n_components=100, random_state=42)
X_tsvd_subset = tsvd_viz.fit_transform(X_subset)

pca2 = PCA(n_components=2, random_state=42)
X_pca2 = pca2.fit_transform(X_tsvd_subset)
```

### The idea:
```
Problem: PCA on sparse 5000 features → failed (1.58%)
New approach:
Step 1: TSVD reduces sparse 5000 → dense 100
Step 2: PCA reduces dense 100 → 2 for plotting
```

### Why it still failed:
```
Text clusters are NON-LINEAR!
Categories don't separate along straight lines —
they separate along curves!

PCA (even after TSVD) = only straight line directions
= still can't find curved boundaries ❌

Need non-linear method! → t-SNE ✅
```

---

## ✨ Cell 28 — TSVD → t-SNE (Success!)

```python
tsne = TSNE(n_components=2,
            random_state=42,
            perplexity=30,
            max_iter=1000)

X_tsne = tsne.fit_transform(X_tsvd_subset)
```

### What is t-SNE?
t-SNE = t-distributed Stochastic Neighbor Embedding

Simple explanation:
```
For each article, t-SNE asks:
"Which other articles are most similar to me?"

Then arranges articles in 2D so:
→ Similar articles stay CLOSE together
→ Different articles stay FAR apart

Non-linear — uses curves, not straight lines! ✅
```

### Parameters:

**`perplexity=30`**
Most important t-SNE parameter. Controls how many "neighbors" each point considers:
```
Low (5)  → very local structure → tight small clusters
High (50) → global structure → larger spread clusters
30 = sweet spot for 5000+ points ✅
```

**`max_iter=1000`**
t-SNE learns by iteratively improving positions. 1000 steps = good balance of quality vs speed.

### Why t-SNE is slow (2-3 minutes):
```
For each of 5567 articles:
Calculate similarity with ALL others
5567 × 5567 = 31,000,000 comparisons!
Done 1000 times = 31 BILLION operations!

Worth the wait — results are beautiful! ✅
```

### Why TSVD first then t-SNE:
```python
# t-SNE directly on 5000 features:
# 5567 × 5567 × 5000 = CRASHES! 💥

# TSVD first reduces to 100 features:
# 5567 × 5567 × 100 = manageable! ✅

# TSVD → t-SNE = industry standard NLP visualization!
```

### PCA vs t-SNE comparison:

| | PCA | t-SNE |
|---|---|---|
| Method | Linear | Non-linear |
| Speed | Very fast | Slow (minutes) |
| Preserves | Global variance | Local neighborhoods |
| Text clusters | ❌ Fails | ✅ Beautiful! |

---

## 🔪 Cell 29 — Train Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_tsvd_subset,
    y_subset,
    test_size=0.2,
    random_state=42,
    stratify=y_subset
)
```

### Why `X_tsvd_subset` not `X_subset`?
```
X_subset     = sparse TF-IDF (5567 × 5000) → too many features!
X_tsvd_subset = dense TSVD (5567 × 100)   → reduced, clean ✅

Benefits of using TSVD output:
1. Already reduced noise
2. Dense matrix — works with all classifiers
3. 100 features instead of 5000 → faster training!
```

### The `stratify` parameter — most important!
```
WITHOUT stratify → random split:
Train → 900 hockey, 200 space, 800 religion...
Test  → 99 hockey, 790 space, 128 religion...
Completely unbalanced! ❌

WITH stratify=y_subset:
Each category maintains its proportion:
If 18% of data is hockey:
Train → 18% hockey ✅
Test  → 18% hockey ✅
Fair evaluation! ✅
```

### Why same split matters for comparison:
All three models (LDA, SVM, NMF+SVM) must use the SAME test set for fair comparison. Same `random_state=42` guarantees this!

---

## 🏆 Cell 30 — LDA Classification

```python
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
y_pred_lda = lda.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_test, y_pred_lda) * 100, 2)}%")
print(classification_report(y_test, y_pred_lda,
      target_names=selected_categories))
```

### What LDA does internally during `fit()`:
```
Step 1: Calculate mean of each class
Step 2: Calculate between-class scatter (how far apart are means?)
Step 3: Calculate within-class scatter (how spread is each class?)
Step 4: Find directions maximizing between/within ratio
= maximum class separation! 🎯
```

### Classification report metrics:

**Precision:**
```
Of all articles predicted as "space"
how many were actually space?

Precision = True Positives / (True Positives + False Positives)

sci.space precision = 0.81:
100 predicted as space → 81 actually space ✅, 19 wrong ❌
```

**Recall:**
```
Of all actual space articles
how many did we correctly find?

Recall = True Positives / (True Positives + False Negatives)

sci.space recall = 0.95:
195 actual space articles → found 185 ✅, missed 10 ❌
```

**F1-Score:**
```
Balance between precision and recall
F1 = 2 × (Precision × Recall) / (Precision + Recall)

Use F1 when classes are imbalanced
(religion has 628, hockey has 999)
F1 gives fairer picture! ✅
```

**Support:**
```
Number of actual articles of that category in test set
sci.space support = 195 test articles
```

### Key findings from our results:
```
talk.religion.misc precision = 1.00! 🏆
→ Every article predicted as religion WAS religion!
→ Religion uses unique vocabulary nobody else uses
  (god, jesus, bible, church, faith)

sci.space precision = 0.81 (lowest)
→ Some non-space articles mislabeled as space
→ Probably sci.med articles mentioning
  "space" in medical context!
```

---

## 📊 Cell 31 — Final Comparison

```python
# Model 1 - LDA (already trained)
lda_acc = round(accuracy_score(y_test, y_pred_lda) * 100, 2)

# Model 2 - TSVD + SVM
svm = LinearSVC(random_state=42, max_iter=2000)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
svm_acc = round(accuracy_score(y_test, y_pred_svm) * 100, 2)

# Model 3 - NMF + SVM
X_nmf_subset = X_nmf[mask.values]
X_train_nmf, X_test_nmf, _, _ = train_test_split(
    X_nmf_subset, y_subset,
    test_size=0.2, random_state=42, stratify=y_subset
)
svm_nmf = LinearSVC(random_state=42, max_iter=2000)
svm_nmf.fit(X_train_nmf, y_train)
y_pred_nmf = svm_nmf.predict(X_test_nmf)
nmf_acc = round(accuracy_score(y_test, y_pred_nmf) * 100, 2)
```

### Key code concepts:

**`_, _` (underscore variables)**
```python
X_train_nmf, X_test_nmf, _, _ = train_test_split(...)

# train_test_split returns 4 values
# We already have y_train and y_test from Cell 29!
# _ = Python convention for "I don't need this value"
# Ignore y values, only need X_train_nmf and X_test_nmf ✅
```

**`X_nmf[mask.values]`**
```python
# X_nmf computed on FULL dataset (18846 × 20)
# Need only our 6-category subset!
# mask.values = [True, False, True, ...]
# Result: (5567 × 20) ✅
```

**`max_iter=2000` for LinearSVC**
```python
# SVM learns iteratively
# Default 1000 sometimes not enough for text!
# 2000 ensures full convergence ✅
```

### Bar chart tricks:

**`zip(bars, accuracies)`**
```python
# Combines two lists together:
# (bar1, 91.2), (bar2, 93.27), (bar3, 81.6)
# Used to add accuracy label on top of each bar!
```

**`bar.get_x() + bar.get_width()/2`**
```python
# get_x()       → left edge of bar
# get_width()/2 → half width
# Together      → CENTER of bar for text placement! ✅
```

**`plt.ylim(80, 100)`**
```python
# Y axis from 80 to 100 (not 0 to 100!)
# Without this: 93% and 91% bars look identical!
# Zooming in makes differences clearly visible! ✅
```

---

## 🏆 Final Results

| Technique | Dataset | Accuracy |
|---|---|---|
| TSVD + SVM | 6 categories | **93.27%** 🥇 |
| TSVD + LDA | 6 categories | 91.2% 🥈 |
| NMF + SVM | 6 categories | 81.6% 🥉 |
| TSVD + SVM | 20 categories | 83.24% |

### Why these results make sense:

**TSVD + SVM wins:**
```
TSVD → best dimensionality reduction for sparse text
SVM  → best classifier for high dimensional text
Together → industry standard NLP pipeline! ✅
```

**NMF loses:**
```
NMF was designed for TOPIC DISCOVERY not classification!
20 NMF topics = too little information for classification
Wrong tool for this job! ❌
```

**20 categories harder than 6:**
```
Similar categories confuse model:
comp.sys.ibm.pc.hardware vs comp.sys.mac.hardware
→ both talk about computers, hardware, memory

Random chance for 6 cats  = 16.7%
Random chance for 20 cats = 5%
Our model: 83.24% = still 16x better than random! 💪
```

---

## 🎓 Key Takeaways — Interview Ready!

```
1. PCA fails on sparse text (1.58% variance) — use TSVD instead!

2. TSVD is the right tool for NLP because:
   → Works on sparse matrices natively
   → No centering needed (preserves sparsity)
   → Fast and memory efficient

3. NMF discovers meaningful interpretable topics
   → No labels needed (unsupervised)
   → All components are positive (interpretable!)
   → Great for topic modeling, NOT classification

4. Text clusters are non-linear
   → PCA fails at visualization
   → t-SNE works perfectly (neighborhood-based)

5. TSVD + SVM = industry standard NLP classification pipeline
   → Used in production by major tech companies

6. Always use stratify in train_test_split for classification!
   → Ensures balanced representation of all classes
```

---

## 🔄 Complete Pipeline Summary

```
Raw text (18,846 articles)
        ↓ regex cleaning (Cell 10-11)
        ↓ TF-IDF vectorization (Cell 15-17) → (18846 × 5000 sparse)
        ↓
   ┌────┴────┐
   │         │
TSVD(300)   NMF(20)
(18846×300) (18846×20)
   │         │
   │         └── Topic Discovery ✅ (19/20 topics meaningful!)
   │
   ├── Full dataset SVM → 83.24% (20 categories)
   │
   └── Subset (5567 × 6 categories)
            │
       ┌────┴────┐
       │         │
   TSVD(100)  NMF subset
       │         │
       ├── PCA(2)      → 1.58% variance ❌ blob
       ├── KernelPCA(2) → partial separation ⚠️
       ├── t-SNE(2)    → beautiful clusters ✅
       │
       ├── LDA → 91.2% accuracy 🥈
       └── SVM → 93.27% accuracy 🥇 WINNER!
```

---

*Written as personal learning notes while building NewsAnalyzer* 📝
*Dataset: 20 Newsgroups — sklearn built-in*
*Author: Avik Sarkar*
