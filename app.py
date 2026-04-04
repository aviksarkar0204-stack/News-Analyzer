import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF, PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.manifold import TSNE

# ── PAGE CONFIG ──────────────────────────────────────
st.set_page_config(
    page_title="NewsAnalyzer",
    page_icon="📰",
    layout="wide"
)

# ── STYLING ──────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #0d1b2a; color: #e0e0e0; }
h1, h2, h3 { color: #4fc3f7; }
.metric-card {
    background-color: #1a2a3a;
    border: 1px solid #4fc3f7;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    margin-bottom: 10px;
}
.metric-value { font-size: 28px; font-weight: bold; color: #4fc3f7; }
.metric-label { font-size: 13px; color: #90caf9; }
.result-card {
    background-color: #1a2a3a;
    border-radius: 12px;
    padding: 25px;
    text-align: center;
    margin-top: 15px;
}
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ─────────────────────────────────────────
SELECTED_CATEGORIES = [
    'sci.space', 'rec.sport.hockey',
    'talk.religion.misc', 'comp.graphics',
    'rec.autos', 'sci.med'
]
COLORS = ['#e74c3c', '#3498db', '#2ecc71',
          '#f39c12', '#9b59b6', '#1abc9c']

# ── HELPER FUNCTIONS ──────────────────────────────────
def clean_text(text):
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def make_plot_dark(fig, ax):
    fig.patch.set_facecolor('#0d1b2a')
    ax.set_facecolor('#0d1b2a')
    ax.tick_params(colors='#90caf9')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1a2a3a')
    ax.grid(True, alpha=0.1, color='#4fc3f7')

# ── MAIN TRAINING FUNCTION ────────────────────────────
@st.cache_resource
def load_and_train():

    # Step 1 — Load data
    data = fetch_20newsgroups(subset='all')
    df = pd.DataFrame({
        'text': data.data,
        'label': data.target,
        'category': [data.target_names[i] for i in data.target]
    })
    df['cleaned_text'] = df['text'].apply(clean_text)
    df.drop(columns=['text'], inplace=True)

    # Step 2 — TF-IDF
    tfidf = TfidfVectorizer(
        max_features=5000, stop_words='english',
        min_df=5, max_df=0.95
    )
    X = tfidf.fit_transform(df['cleaned_text'])
    y = df['label'].values
    feature_names = tfidf.get_feature_names_out()
    target_names = data.target_names

    # Step 3 — TSVD on full dataset
    tsvd_full = TruncatedSVD(n_components=300, random_state=42)
    X_tsvd_full = tsvd_full.fit_transform(X)

    # Step 4 — NMF on full dataset
    nmf = NMF(n_components=20, random_state=42, max_iter=400)
    X_nmf = nmf.fit_transform(X)

    # Step 5 — Subset for visualization & classification
    mask = df['category'].isin(SELECTED_CATEGORIES)
    df_subset = df[mask].reset_index(drop=True)
    X_subset = X[mask.values]
    y_subset = df_subset['label'].values
    X_nmf_subset = X_nmf[mask.values]

    # Step 6 — TSVD on subset (for visualization)
    tsvd_viz = TruncatedSVD(n_components=100, random_state=42)
    X_tsvd_subset = tsvd_viz.fit_transform(X_subset)

    # Step 7 — Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_tsvd_subset, y_subset,
        test_size=0.2, random_state=42,
        stratify=y_subset
    )
    X_train_nmf, X_test_nmf, _, _ = train_test_split(
        X_nmf_subset, y_subset,
        test_size=0.2, random_state=42,
        stratify=y_subset
    )

    # Step 8 — Train LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    lda_acc = round(accuracy_score(y_test, lda.predict(X_test)) * 100, 2)

    # Step 9 — Train SVM (TSVD features)
    svm = LinearSVC(random_state=42, max_iter=2000)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    svm_acc = round(accuracy_score(y_test, y_pred_svm) * 100, 2)

    # Step 10 — Train SVM (NMF features)
    svm_nmf = LinearSVC(random_state=42, max_iter=2000)
    svm_nmf.fit(X_train_nmf, y_train)
    nmf_acc = round(accuracy_score(
        y_test, svm_nmf.predict(X_test_nmf)) * 100, 2)

    # Step 11 — Classification report
    report = classification_report(
        y_test, y_pred_svm,
        target_names=SELECTED_CATEGORIES,
        output_dict=True
    )

    return {
        # Data
        'df': df,
        'df_subset': df_subset,
        'target_names': target_names,
        # Features
        'X_subset': X_subset,
        'X_tsvd_subset': X_tsvd_subset,
        'y_subset': y_subset,
        # Models (reused for prediction!)
        'tfidf': tfidf,
        'tsvd_viz': tsvd_viz,
        'svm': svm,
        'nmf': nmf,
        # Meta
        'feature_names': feature_names,
        'X_train': X_train,
        'y_train': y_train,
        # Results
        'lda_acc': lda_acc,
        'svm_acc': svm_acc,
        'nmf_acc': nmf_acc,
        'report': report,
    }

# ── t-SNE (separate cache — slow!) ───────────────────
@st.cache_resource
def compute_tsne(_X_tsvd_subset):
    tsne = TSNE(n_components=2, random_state=42,
                perplexity=30, max_iter=1000)
    return tsne.fit_transform(_X_tsvd_subset)

# ── SIDEBAR ───────────────────────────────────────────
st.sidebar.title("📰 NewsAnalyzer")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "🔍 Topic Discovery",
    "🌐 Visualization",
    "🤖 Classification"
])
st.sidebar.markdown("---")
st.sidebar.caption("Dataset: 20 Newsgroups (sklearn)")
st.sidebar.caption("Articles: 18,846 | Categories: 20")

# ── LOAD EVERYTHING ───────────────────────────────────
with st.spinner("⏳ Loading and training models... first load takes ~2 minutes"):
    R = load_and_train()

# Unpack everything cleanly
df            = R['df']
df_subset     = R['df_subset']
target_names  = R['target_names']
X_subset      = R['X_subset']
X_tsvd_subset = R['X_tsvd_subset']
y_subset      = R['y_subset']
tfidf         = R['tfidf']
tsvd_viz      = R['tsvd_viz']
svm           = R['svm']
nmf           = R['nmf']
feature_names = R['feature_names']
X_train       = R['X_train']
y_train       = R['y_train']
lda_acc       = R['lda_acc']
svm_acc       = R['svm_acc']
nmf_acc       = R['nmf_acc']
report        = R['report']

# ════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("📰 NewsAnalyzer")
    st.markdown("#### Intelligent News Classification & Topic Discovery")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in zip(
        [c1, c2, c3, c4],
        ["18,846", "20", f"{svm_acc}%", "5"],
        ["Total Articles", "Categories", "Best Accuracy", "Techniques Used"]
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📊 Category Distribution")

    fig, ax = plt.subplots(figsize=(14, 5))
    make_plot_dark(fig, ax)
    counts = df['category'].value_counts()
    ax.bar(counts.index, counts.values, color='#4fc3f7', alpha=0.85)
    ax.set_xlabel('Category', color='#90caf9')
    ax.set_ylabel('Articles', color='#90caf9')
    ax.tick_params(axis='x', rotation=90)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("🛠️ Techniques Comparison Table")
    tech_df = pd.DataFrame({
        'Technique': ['PCA', 'TSVD', 'NMF', 'Kernel PCA', 't-SNE', 'LDA', 'SVM'],
        'Type': ['Linear', 'Linear', 'Non-negative', 'Non-linear',
                 'Non-linear', 'Supervised', 'Supervised'],
        'Purpose': ['Visualization', 'Dim. Reduction', 'Topic Discovery',
                    'Visualization', 'Visualization', 'Classification', 'Classification'],
        'Result': [
            '❌ 1.58% variance', '✅ 30.88% variance', '✅ 20 topics found',
            '⚠️ Partial clusters', '✅ Clear clusters',
            f'✅ {lda_acc}% accuracy', f'🏆 {svm_acc}% accuracy'
        ]
    })
    st.dataframe(tech_df, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════
# PAGE 2 — TOPIC DISCOVERY
# ════════════════════════════════════════════════════
elif page == "🔍 Topic Discovery":
    st.title("🔍 Topic Discovery — NMF")
    st.markdown("NMF automatically discovers **20 hidden topics** without any labels!")
    st.markdown("---")

    n_words = st.slider("Top words per topic", 5, 15, 10)
    st.markdown("---")

    for topic_idx, topic in enumerate(nmf.components_):
        top_words = [feature_names[i]
                     for i in topic.argsort()[-n_words:][::-1]]
        with st.expander(f"📌 Topic {topic_idx + 1}", expanded=topic_idx < 3):
            st.markdown(" | ".join([f"`{w}`" for w in top_words]))

# ════════════════════════════════════════════════════
# PAGE 3 — VISUALIZATION
# ════════════════════════════════════════════════════
elif page == "🌐 Visualization":
    st.title("🌐 Article Cluster Visualization")
    st.markdown("Visualizing **6 news categories** using different techniques.")
    st.markdown("---")

    viz_option = st.selectbox("Select Visualization:", [
        "PCA (Linear — baseline)",
        "TSVD → PCA (Improved linear)",
        "TSVD → t-SNE (Best!)"
    ])

    fig, ax = plt.subplots(figsize=(12, 7))
    make_plot_dark(fig, ax)

    if viz_option == "PCA (Linear — baseline)":
        with st.spinner("Running PCA..."):
            pca = PCA(n_components=2, random_state=42)
            X_viz = pca.fit_transform(X_subset.toarray())
            variance = round(pca.explained_variance_ratio_.sum() * 100, 2)
        st.warning(f"⚠️ PCA retained only **{variance}%** variance — poor for text!")

    elif viz_option == "TSVD → PCA (Improved linear)":
        with st.spinner("Running TSVD → PCA..."):
            pca2 = PCA(n_components=2, random_state=42)
            X_viz = pca2.fit_transform(X_tsvd_subset)
        st.warning("⚠️ Still linear — text clusters are non-linear!")

    else:
        with st.spinner("Running t-SNE... (2-3 minutes) ☕ Cached after first run!"):
            X_viz = compute_tsne(X_tsvd_subset)
        st.success("✅ t-SNE gives the best cluster separation!")

    for i, category in enumerate(SELECTED_CATEGORIES):
        mask_cat = df_subset['category'] == category
        ax.scatter(
            X_viz[mask_cat, 0], X_viz[mask_cat, 1],
            c=COLORS[i], label=category, alpha=0.6, s=10
        )

    ax.legend(markerscale=3, facecolor='#1a2a3a',
              labelcolor='#e0e0e0', fontsize=9)
    ax.set_xlabel("Component 1", color='#90caf9')
    ax.set_ylabel("Component 2", color='#90caf9')
    ax.set_title(viz_option, color='#4fc3f7', fontsize=13)
    plt.tight_layout()
    st.pyplot(fig)

# ════════════════════════════════════════════════════
# PAGE 4 — CLASSIFICATION
# ════════════════════════════════════════════════════
elif page == "🤖 Classification":
    st.title("🤖 Classification Results")
    st.markdown("Comparing **3 classification pipelines** on 6 news categories.")
    st.markdown("---")

    # Metric cards
    c1, c2, c3 = st.columns(3)
    for col, val, label in zip(
        [c1, c2, c3],
        [f"{lda_acc}%", f"{svm_acc}%", f"{nmf_acc}%"],
        ["TSVD + LDA", "TSVD + SVM 🏆", "NMF + SVM"]
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📊 Accuracy Comparison")

    fig, ax = plt.subplots(figsize=(8, 5))
    make_plot_dark(fig, ax)
    techniques = ['TSVD + LDA', 'TSVD + SVM', 'NMF + SVM']
    accuracies = [lda_acc, svm_acc, nmf_acc]
    bar_colors = ['#4fc3f7', '#2ecc71', '#e74c3c']
    bars = ax.bar(techniques, accuracies, color=bar_colors, width=0.4)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f'{acc}%', ha='center',
                fontweight='bold', fontsize=12, color='white')
    ax.set_ylim(75, 100)
    ax.set_ylabel('Accuracy (%)', color='#90caf9')
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("📋 Detailed Report (TSVD + SVM)")
    report_df = pd.DataFrame(report).transpose().round(2)
    report_df = report_df[report_df.index.isin(SELECTED_CATEGORIES)]
    report_df = report_df[['precision', 'recall', 'f1-score', 'support']]
    report_df.columns = ['Precision', 'Recall', 'F1-Score', 'Support']
    st.dataframe(report_df, use_container_width=True)

    st.markdown("---")
    st.subheader("🧠 Why TSVD + SVM Wins?")
    st.markdown("""
    - **TSVD** is best for sparse text — works on sparse matrices natively
    - **SVM** finds optimal decision boundaries in high dimensional space
    - Together = **industry standard NLP classification pipeline!**
    - NMF was designed for topic discovery — not classification
    - LDA is limited to `n_classes - 1 = 5` directions only
    """)

    # ── ARTICLE ANALYZER ─────────────────────────────
    st.markdown("---")
    st.subheader("🔎 Analyze Your Own Article!")
    st.markdown("Paste any news article and our model will predict its category!")

    user_article = st.text_area(
        "Paste your article here:",
        height=200,
        placeholder="e.g. NASA launched a new shuttle mission today targeting the Moon..."
    )

    if st.button("🔍 Analyze Article", use_container_width=True):
        if user_article.strip() == "":
            st.warning("⚠️ Please paste an article first!")
        else:
            with st.spinner("Analyzing..."):
                # Clean → TF-IDF → TSVD → Predict
                cleaned = clean_text(user_article)
                X_user = tfidf.transform([cleaned])
                X_user_tsvd = tsvd_viz.transform(X_user)
                prediction = svm.predict(X_user_tsvd)[0]
                predicted_category = target_names[prediction]

            # Result card
            if predicted_category in SELECTED_CATEGORIES:
                color = COLORS[SELECTED_CATEGORIES.index(predicted_category)]
            else:
                color = "#4fc3f7"

            st.markdown(f"""
            <div class="result-card" style="border: 2px solid {color};">
                <div style="font-size:14px; color:#90caf9; margin-bottom:8px;">
                    Predicted Category
                </div>
                <div style="font-size:36px; font-weight:bold; color:{color};">
                    {predicted_category}
                </div>
                <div style="font-size:12px; color:#90caf9; margin-top:8px;">
                    Model: TSVD + LinearSVC
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")

            # Keywords detected
            st.markdown("**🔑 Key words detected:**")
            vocab = set(feature_names)
            keywords = list(set([
                w for w in cleaned.split() if w in vocab
            ]))[:20]
            if keywords:
                st.markdown(" ".join([f"`{w}`" for w in keywords]))
            else:
                st.info("No significant keywords found.")
