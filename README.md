# 📰 NewsAnalyzer

> A machine learning powered news article analysis and classification system built with Python, sklearn, and Streamlit.

> ⚠️ **Practice Project — Work in Progress** — This project is incomplete and built purely for **learning and practice purposes**. It was created to explore and compare feature extraction and dimensionality reduction techniques (PCA, TSVD, NMF, Kernel PCA, t-SNE) on real text data. It is not a production-ready application and results should not be used for any real world decisions.

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://news-analyzer-wvaizwm4qn2xew5uaiwvgm.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![sklearn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

---

## 📌 Project Overview

NewsAnalyzer is a data science portfolio project that demonstrates the complete NLP pipeline — from raw text cleaning to dimensionality reduction, topic discovery, and classification. It applies **5 different dimensionality reduction techniques** on the same dataset and compares their effectiveness side by side.

The project uses the **20 Newsgroups dataset** — a collection of 18,846 real news articles across 20 categories — to build an intelligent news analysis system that automatically discovers topics, visualizes article clusters, and classifies articles into categories with **93.27% accuracy.**

---

## 🚀 Live Demo

> 🔗 **[Click here to open the app](https://news-analyzer-wvaizwm4qn2xew5uaiwvgm.streamlit.app/)**

> ⚠️ First load may take 2-3 minutes as models train on startup. Subsequent loads are instant thanks to Streamlit caching!

---

## 🖥️ App Features

### 🏠 Page 1 — Overview
- Key stats: total articles, categories, best accuracy
- Category distribution bar chart
- Full technique comparison table

### 🔍 Page 2 — Topic Discovery
- NMF automatically discovers 20 hidden topics from 18,846 articles
- Interactive slider to control number of words shown per topic
- Expandable topic cards showing top keywords

### 🌐 Page 3 — Visualization
- Compare 3 different visualization approaches:
  - PCA (linear baseline — shows why it fails on text)
  - TSVD → PCA (improved but still limited)
  - TSVD → t-SNE (best — beautiful cluster separation!)
- Interactive dropdown to switch between techniques

### 🤖 Page 4 — Classification
- Accuracy comparison of 3 classification pipelines
- Detailed classification report (Precision, Recall, F1)
- **Live Article Analyzer** — paste any article and get instant category prediction!

---

## 🗂️ Dataset

| Property | Details |
|---|---|
| **Name** | 20 Newsgroups |
| **Source** | Built into sklearn (`fetch_20newsgroups`) |
| **Total Articles** | 18,846 |
| **Categories** | 20 |
| **Type** | Text (news articles) |
| **Download** | Automatic (~14MB, cached after first download) |

### Categories Include:
- **Sports** → rec.sport.hockey, rec.sport.baseball
- **Technology** → comp.graphics, comp.os.ms-windows, comp.sys.hardware
- **Science** → sci.space, sci.med, sci.electronics
- **Politics** → talk.politics.guns, talk.politics.misc
- **Religion** → talk.religion.misc, soc.religion.christian, alt.atheism
- **Misc** → rec.autos, rec.motorcycles, misc.forsale

---

## 🧠 ML Pipeline

```
Raw Text (18,846 articles)
        ↓ Text Cleaning (regex)
        ↓ TF-IDF Vectorization → (18846 × 5000 sparse matrix)
        ↓
   ┌────┴────┐
   │         │
TSVD(300)  NMF(20)
   │         │
   │         └── Topic Discovery (20 meaningful topics!)
   │
   └── Subset (5567 articles, 6 categories)
            │
       ┌────┴────────────┐
       │                 │
   TSVD(100)         NMF subset
       │                 │
       ├── PCA(2)         → ❌ 1.58% variance (fails!)
       ├── Kernel PCA(2)  → ⚠️ partial clusters
       ├── t-SNE(2)       → ✅ beautiful clusters!
       │
       ├── LDA → 91.2% accuracy
       └── SVM → 93.27% accuracy 🏆
```

---

## 📊 Results

### Dimensionality Reduction Comparison

| Technique | Type | Purpose | Result |
|---|---|---|---|
| **PCA** | Linear | Visualization | ❌ 1.58% variance — fails on text |
| **TSVD** | Linear | Dim. Reduction | ✅ 30.88% variance retained |
| **NMF** | Non-negative | Topic Discovery | ✅ 20 meaningful topics found |
| **Kernel PCA** | Non-linear | Visualization | ⚠️ Partial cluster separation |
| **t-SNE** | Non-linear | Visualization | ✅ Beautiful clear clusters! |

### Classification Comparison

| Pipeline | Accuracy |
|---|---|
| **TSVD + SVM** | 🏆 **93.27%** |
| **TSVD + LDA** | 91.2% |
| **NMF + SVM** | 81.6% |
| **TSVD + SVM (20 categories)** | 83.24% |

### NMF Topic Discovery (Sample)

```
Topic  1: god | jesus | bible | christian | faith    → Religion ✅
Topic  3: game | team | hockey | baseball | players  → Sports ✅
Topic  4: key | clipper | encryption | crypto | keys → Cryptography ✅
Topic 10: car | engine | dealer | ford | oil         → Automobiles ✅
Topic 15: space | shuttle | nasa | launch | moon     → Space ✅
```

19 out of 20 topics perfectly matched real categories — without any labels! 🤩

---

## 🔁 Extended Use — SGD Online Learning

The 20 Newsgroups dataset was later reused in a separate practice project — **SGD StreamText** — to explore **online learning** using `SGDClassifier` with `partial_fit()`.

Instead of training on the full dataset at once, articles were fed in mini-batches of 500 and the model updated itself incrementally — simulating a real-world data stream. The learning curve showed accuracy rising from 0.65 after the first batch to 0.86 after all batches, closely approaching the `LogisticRegression` batch baseline of 0.87.

> 📄 See the separate `SGD_StreamText` repository for the full notebook and results.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.10+ |
| **ML Library** | scikit-learn |
| **Web App** | Streamlit |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib |
| **Text Processing** | sklearn TfidfVectorizer, regex |
| **Dimensionality Reduction** | PCA, TSVD, NMF, Kernel PCA, t-SNE |
| **Classification** | LinearSVC, LinearDiscriminantAnalysis |

---

## 📁 Project Structure

```
NewsAnalyzer/
├── app.py                  ← Main Streamlit application
├── news_analyzer.ipynb     ← Full EDA & model building notebook
├── EXPLANATION.md          ← Detailed code explanation (cell by cell)
├── requirements.txt        ← Python dependencies
└── README.md               ← Project documentation
```

---

## ⚙️ Run Locally

**1. Clone the repository:**
```bash
git clone YOUR_GITHUB_REPO_URL_HERE
cd NewsAnalyzer
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run the app:**
```bash
streamlit run app.py
```

**4. First load:**
- Downloads 20 Newsgroups dataset (~14MB) automatically
- Trains all models (~2 minutes)
- All subsequent loads are instant (cached!)

---

## 📋 Requirements

```
streamlit
scikit-learn
pandas
numpy
matplotlib
```

---

## 🔍 Key Learnings

**1. PCA fails on sparse text data**
Text TF-IDF matrices are extremely sparse — PCA could only retain 1.58% variance with 2 components. TSVD handles sparse matrices natively and is always preferred for NLP.

**2. NMF discovers interpretable topics**
Unlike PCA whose components are hard to interpret, NMF components are additive and positive — making them directly readable as topics. Without any labels it discovered all 20 newsgroup topics.

**3. Text clusters are non-linear**
Linear techniques (PCA, LDA) struggle to visually separate text clusters. Non-linear t-SNE beautifully separated all 6 categories proving text data has curved decision boundaries.

**4. TSVD + SVM = industry standard NLP pipeline**
This combination is used in production by major tech companies for text classification. It balances accuracy, speed, and memory efficiency perfectly.

**5. NMF is wrong for classification**
NMF excels at topic discovery but loses too much information for classification (81.6% vs 93.27%). Always match the technique to the task!

---

## ⚠️ Limitations

- App trains models on startup — first load takes ~2 minutes
- Article analyzer works best for articles similar to the 20 Newsgroups style
- t-SNE visualization takes 2-3 minutes to compute (cached after first run)
- Classification limited to 6 selected categories for the live analyzer

---

## 🔮 Future Improvements

- [ ] Add Kernel PCA to visualization page comparison
- [ ] Expand article analyzer to all 20 categories
- [ ] Add word cloud visualization for each NMF topic
- [ ] Add confidence score to article prediction
- [ ] Try transformer-based models (BERT) for higher accuracy
- [ ] Add upload CSV feature for batch article classification

---

## 👨‍💻 Author

**Avik Sarkar**
- GitHub: [@aviksarkar0204-stack](YOUR_GITHUB_PROFILE_URL_HERE)
- Dataset: [20 Newsgroups — sklearn](https://scikit-learn.org/stable/datasets/real_world.html#newsgroups-dataset)

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).

---

*Built with ❤️ as part of a self-directed ML learning journey 🚀*
*Part of a portfolio series: MandiPredict → IPL Dashboard → NewsAnalyzer*
