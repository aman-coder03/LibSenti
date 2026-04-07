import streamlit as st
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
from PIL import Image
import os
import pandas as pd

# === Load Model & Tokenizer ===
model_path = r'coder0304/libsenti-roberta-sentiment'
@st.cache_resource
def load_model():
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    return tokenizer, model, device

tokenizer, model, device = load_model()

# === Predict Sentiment Function ===
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)

    negative_prob = probs[0][0].item()
    neutral_prob = probs[0][1].item()
    positive_prob = probs[0][2].item()

    predicted_class = torch.argmax(probs, dim=1).item()

    label_map = {
        0: "Negative",
        1: "Neutral",
        2: "Positive"
    }

    sentiment = label_map[predicted_class]

    confidence = max(negative_prob, neutral_prob, positive_prob)

    return sentiment, confidence, {'Negative': negative_prob, 'Neutral': neutral_prob, 'Positive': positive_prob}

# Helper to sanitize display name to filename with suffix
def filename_from_display_name(name, suffix):
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
    return f"{safe_name}_{suffix}.png"

# === Streamlit UI ===
st.set_page_config(page_title="LibSenti", page_icon="📚", layout="wide")
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
<style>
    html, body, .stApp, .block-container {
        background-color: #1c1c1c;
        color: #f0f0f0 !important;
        font-family: 'Inter', sans-serif !important;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #2d2d2d;
        padding: 1rem 2rem;
        margin-right: 0.5rem;
        border: 1px solid #555;
        border-radius: 10px;
        box-shadow: none;
        color: #f0f0f0;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #3c3c3c;
        transition: 0.2s;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #404040;
        border: 1px solid #888;
    }
    .stTextArea textarea,
    .stSelectbox div[data-baseweb="select"],
    .stExpander,
    .st-c5,
    .st-c6 {
        background-color: #2a2a2a !important;
        color: #f0f0f0;
        border-radius: 10px;
        border: 1px solid #555;
    }
    .stProgress > div > div > div {
        background-color: #6c91bf;
    }
    .stCheckbox > label,
    .stRadio > label,
    label,
    p,
    .stMarkdown,
    .stCaption,
    .css-1v0mbdj {
        color: #f0f0f0 !important;
    }
</style>
""", unsafe_allow_html=True)

# === Title ===
st.markdown("""
    <h1 style='text-align: center; font-size: 3.5rem; color: white; margin-bottom: 2.5rem;'>
        📚 LibSenti
    </h1>
""", unsafe_allow_html=True)

# === Tabs as Cards ===
st.markdown("<div class='tabs-card-container'>", unsafe_allow_html=True)
tabs = st.tabs([
    "\U0001F3E0 Home",
    "\U0001F4DD Sentiment Predictor",
    "\U0001F524 Unigram WordClouds",
    "\U0001F517 Bigram WordClouds",
    "\U0001F50D Pie Chart Comparison",
    "\U0001F4CA IIT vs NIT Chart",
    "\U0001F31F Library Experiences"
])
st.markdown("</div>", unsafe_allow_html=True)

# === Tab 0: Home ===
with tabs[0]:
    st.markdown("""
    <style>
        .libsenti-container {
            background: transparent;
            padding: 2rem 3rem;
            border-radius: 1rem;
        }
        .libsenti-title {
            font-size: 3.4rem;
            color: #ffffff;
            font-weight: 700;
            margin-bottom: 0.6rem;
        }
        .libsenti-subtitle {
            font-size: 1.6rem;
            color: #dddddd;
            font-weight: 500;
            margin-bottom: 1.2rem;
        }
        .libsenti-description {
            font-size: 1.15rem;
            color: #cccccc;
            margin-bottom: 1.5rem;
        }
        .libsenti-list li {
            font-size: 1.2rem;
            color: #cccccc;
            margin-bottom: 0.6rem;
            padding-left: 0.2rem;
        }
        .libsenti-list li:hover {
            color: #66b3ff;
            transform: translateX(5px);
            transition: 0.2s ease-in-out;
        }
    </style>

    <div class='libsenti-container'>
        <div class='libsenti-title'>Welcome to LibSenti</div>
        <div class='libsenti-subtitle'>Explore AI-Powered Insights from IIT & NIT Library Reviews</div>
        <div class='libsenti-description'>
            LibSenti allows you to analyze real user sentiments using advanced Natural Language Processing and Machine Learning.<br><br>
            Get visual insights, sentiment breakdowns, and keyword analysis—all in one application.
        </div>
        <ul class='libsenti-list'>
            <li>📝 Use <strong>Sentiment Predictor</strong> to analyze your own review.</li>
            <li>🔤 View top <strong>Unigram WordClouds</strong> per institute.</li>
            <li>🔗 Discover frequent word pairs with <strong>Bigram WordClouds</strong>.</li>
            <li>🔍 Explore <strong>Pie Chart Comparison</strong> of sentiment shares.</li>
            <li>📈 Check <strong>IIT vs NIT Sentiment Trends</strong>.</li>
            <li>🌟 Dive into <strong>Highlighted Reviews</strong>—both best and worst experiences.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def add_bottom_padding():
    st.markdown("""
        <style>
            html, body, .stApp {
                background-color: #1c1c1c !important;
            }
            .main, .block-container {
                background-color: #1c1c1c !important;
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            }
            .bottom-padding {
                height: 10vh;
                background-color: #1c1c1c;
            }
        </style>
        <div class="bottom-padding"></div>
    """, unsafe_allow_html=True)


# === Tab 1: Sentiment Predictor ===
with tabs[1]:
    add_bottom_padding()

    st.subheader("📝 Sentiment Predictor")
    user_input = st.text_area("Enter your review:", placeholder="Type a library review here...")

    if st.button("🚀 Predict Sentiment"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter a review to classify.")
        else:
            sentiment, confidence, probs = predict_sentiment(user_input)
            if sentiment == 'Negative':
                st.error(f"😡 Predicted Sentiment: **Negative**")
            elif sentiment == 'Neutral':
                st.warning(f"😐 Predicted Sentiment: **Neutral**")
            else:
                st.success(f"😊 Predicted Sentiment: **Positive**")

            st.markdown("#### 🔍 Sentiment Probabilities:")
            for label in ['Negative', 'Neutral', 'Positive']:
                st.write(f"{label}: {probs[label]:.2f}")
                st.progress(probs[label])

# === Tab 2: Unigram WordClouds ===
with tabs[2]:
    add_bottom_padding()
    st.subheader("🔠 Unigram WordClouds")
    wordcloud_dir = "assets/unigram_wordclouds"
    wordcloud_files = [f for f in os.listdir(wordcloud_dir) if f.endswith("_wordcloud.png")]
    display_names = sorted([f.replace("_wordcloud.png", "").replace("_", " ") for f in wordcloud_files])

    inst1 = st.selectbox("📌 Select Institution 1", [""] + display_names, key="uni_inst1")
    inst2 = st.selectbox("📌 Select Institution 2", [""] + display_names, key="uni_inst2")

    if inst1 and inst2:
        col1, col2 = st.columns(2)
        with col1:
            path1 = os.path.join(wordcloud_dir, filename_from_display_name(inst1, "wordcloud"))
            if os.path.exists(path1):
                st.image(path1, caption=f"{inst1} Unigram WordCloud", use_container_width=True)
        with col2:
            path2 = os.path.join(wordcloud_dir, filename_from_display_name(inst2, "wordcloud"))
            if os.path.exists(path2):
                st.image(path2, caption=f"{inst2} Unigram WordCloud", use_container_width=True)

# === Tab 3: Bigram WordClouds ===
with tabs[3]:
    add_bottom_padding()
    st.subheader("🔗 Bigram WordClouds")
    bigram_wordcloud_dir = "assets/bigram_wordclouds"
    bigram_wc_files = [f for f in os.listdir(bigram_wordcloud_dir) if f.endswith("_bigram_wordcloud.png")]
    display_names = sorted([f.replace("_bigram_wordcloud.png", "").replace("_", " ") for f in bigram_wc_files])

    inst1 = st.selectbox("📌 Select Institution 1", [""] + display_names, key="bi_inst1")
    inst2 = st.selectbox("📌 Select Institution 2", [""] + display_names, key="bi_inst2")

    if inst1 and inst2:
        col1, col2 = st.columns(2)
        with col1:
            path1 = os.path.join(bigram_wordcloud_dir, filename_from_display_name(inst1, "bigram_wordcloud"))
            if os.path.exists(path1):
                st.image(path1, caption=f"{inst1} Bigram WordCloud", use_container_width=True)
        with col2:
            path2 = os.path.join(bigram_wordcloud_dir, filename_from_display_name(inst2, "bigram_wordcloud"))
            if os.path.exists(path2):
                st.image(path2, caption=f"{inst2} Bigram WordCloud", use_container_width=True)

# === Tab 4: Pie Chart Comparison ===
with tabs[4]:
    add_bottom_padding()
    st.subheader("🔍 Sentiment Pie Chart Comparison")
    piechart_dir = "assets/piecharts"
    piechart_files = [f for f in os.listdir(piechart_dir) if f.endswith("_piechart.png")]
    pie_display_names = sorted([f.replace("_piechart.png", "").replace("_", " ") for f in piechart_files])

    pie1 = st.selectbox("📌 Select Institution 1", [""] + pie_display_names, key="pie1")
    pie2 = st.selectbox("📌 Select Institution 2", [""] + pie_display_names, key="pie2")

    if pie1 and pie2:
        col1, col2 = st.columns(2)
        with col1:
            pie_path1 = os.path.join(piechart_dir, filename_from_display_name(pie1, "piechart"))
            if os.path.exists(pie_path1):
                st.image(pie_path1, caption=f"{pie1} Sentiment Pie Chart", use_container_width=True)
        with col2:
            pie_path2 = os.path.join(piechart_dir, filename_from_display_name(pie2, "piechart"))
            if os.path.exists(pie_path2):
                st.image(pie_path2, caption=f"{pie2} Sentiment Pie Chart", use_container_width=True)

# === Tab 5: Static IIT vs NIT Chart ===
with tabs[5]:
    add_bottom_padding()
    st.subheader("📊 IIT vs NIT Sentiment Comparison")
    chart_path = "assets/iit_vs_nit_sentiment_comparison.png"
    if os.path.exists(chart_path):
        st.image(Image.open(chart_path), caption="Sentiment Comparison Between IITs and NITs", width=700)
    else:
        st.warning("Comparison chart not found.")

# === Tab 6: Highlighted Reviews ===
with tabs[6]:
    add_bottom_padding()
    st.subheader("🌟 Highlighted Library Experiences")
    show_pos = st.checkbox("😊 Show Interesting Experiences")
    show_neg = st.checkbox("😡 Show Worst Experiences")

    if show_pos or show_neg:
        try:
            df_raw = pd.read_csv("data/raw_reviews.csv", on_bad_lines='skip')
            df_labelled = pd.read_csv("data/sentiment_iit+nit.csv", on_bad_lines='skip')

            df_labelled['label'] = df_labelled['final_sentiment'].map({'Positive': 2, 'Neutral': 1, 'Negative': 0}).fillna(-1).astype(int)
            df_labelled = df_labelled[df_labelled['label'] != -1]

            df_merged = pd.merge(
                df_raw,
                df_labelled[['institution', 'name', 'rating', 'label']],
                on=['institution', 'name', 'rating'],
                how='inner'
            ).dropna(subset=['review_text'])

            if show_pos:
                st.markdown("### 📗 Interesting Experiences")
                for _, row in df_merged[df_merged['label'] == 2].sort_values(by='review_text', key=lambda x: x.str.len(), ascending=False).head(10).iterrows():
                    with st.expander(f"🏛️ {row['institution']}"):
                        st.success(row['review_text'])

            if show_neg:
                st.markdown("### 📕 Worst Experiences")
                for _, row in df_merged[df_merged['label'] == 0].sort_values(by='review_text', key=lambda x: x.str.len(), ascending=False).head(10).iterrows():
                    with st.expander(f"🏛️ {row['institution']}"):
                        st.error(row['review_text'])

        except Exception as e:
            st.error("⚠️ Could not load experiences section.")
            st.text(f"Error: {e}")

# === Footer ===
st.markdown("---")
st.caption("Made by Aman Srivastava | LibSenti - Library Reviews Sentiment Predictor & Analyst")