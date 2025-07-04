# 📚 LibSenti: Library Review Sentiment Predictor & Analyst

**LibSenti** is an end-to-end, AI-powered application that leverages advanced **machine learning** and **natural language processing (NLP)** techniques to classify sentiment polarity — **Positive, Neutral, or Negative** — from student-submitted reviews of **IIT and NIT libraries**. It combines a robust backend sentiment analysis engine with a visually rich, interactive **Streamlit application** for real-time review exploration, data insights, and institutional benchmarking.

---

## 🚀 Key Features

- ✅ **Real-time Review Prediction** *(📝 Sentiment Predictor Tab)*  
  Users can input custom reviews to receive live sentiment predictions with confidence probabilities and visual feedback.

- ✅ **Unigram WordCloud Visualization** *(🔤 Unigram WordClouds Tab)*  
  Generates wordclouds for individual institutions using most frequent single-word terms found in reviews.

- ✅ **Bigram WordCloud Comparison** *(🔗 Bigram WordClouds Tab)*  
  Displays two-institution comparison of most common word pairs (bigrams) extracted from reviews.

- ✅ **Sentiment Pie Chart Comparison** *(🔍 Pie Chart Comparison Tab)*  
  Side-by-side sentiment distribution pie charts for any two selected institutions, including precise percentage labels.

- ✅ **IIT vs NIT Sentiment Analysis** *(📊 IIT vs NIT Chart Tab)*  
  Presents a consolidated sentiment comparison chart contrasting IITs and NITs at a glance.

- ✅ **Library Experience Highlights** *(🌟 Library Experiences Tab)*  
  Displays standout user-submitted reviews—both best and worst experiences—curated by sentiment and length.

---

## 🧠 Model Details

- Architecture: `BERTForSequenceClassification`
- Dataset: IIT & NIT library reviews (labeled Positive, Neutral, Negative)
- Frameworks: `PyTorch`, `Transformers (HuggingFace)`
- Accuracy: **~96% on test data**
- Label Distribution Handling: Threshold-based for confident classification

---

## 📁 Project Structure
LibSenti/
├── assets/
│ ├── wordclouds/ # Wordcloud PNGs for each IIT/NIT
│ └── iit_vs_nit_sentiment_comparison.png
├── saved_model/ # Trained BERT model and tokenizer
├── app.py # Main Streamlit application
├── train_model.py # Script to train the BERT model
├── cleaned_iit+nit_library_reviews.csv
├── sentiment_iit_library_reviews.csv
└── README.md # This file

---

## 🔍 Sentiment Categories

| Label     | Class |
|-----------|-------|
| Negative  | 0     |
| Neutral   | 1     |
| Positive  | 2     |

Class imbalance is addressed using weighted loss during training and probability thresholds during prediction.

---

## 🧠 Model Training

- Base Model: `bert-base-uncased`
- Framework: Hugging Face Transformers + PyTorch
- Strategy: Fine-tuned with weighted cross-entropy loss for class imbalance
- Threshold logic for better handling of imbalanced classes
- Trained using `Trainer` API with evaluation metrics and early stopping

Run training script:
```bash
python train_model.py
```
This script handles:

- Preprocessing
- Tokenization
- Model fine-tuning
- Class weight balancing
- Model saving to ./saved_model/
  
---

## 🎨 Streamlit Application
Launch the Application:
```
streamlit run app.py
```

### 🧩 Components:

- 📥 **Review Classifier** *(📝 Sentiment Predictor Tab)*  
  Enter any library review and instantly receive a sentiment prediction (Positive, Neutral, Negative).

- 📈 **Sentiment Probabilities** *(📝 Sentiment Predictor Tab)*  
  Visualize the confidence scores for each sentiment using interactive progress bars to assess prediction certainty.

- ☁️ **WordCloud Comparator**  
  - 🔤 *(Unigram WordClouds Tab)*: Select and compare two institutions to explore most frequent individual keywords.  
  - 🔗 *(Bigram WordClouds Tab)*: Compare most common two-word combinations to find phrase patterns in reviews.

- 📊 **Sentiment Pie Chart Comparison** *(🔍 Pie Chart Comparison Tab)*  
  Instantly loads sentiment distribution charts for selected institutions side-by-side for intuitive visual analysis.

- 🧮 **IIT vs NIT Overall Chart** *(📊 IIT vs NIT Chart Tab)*  
  A comparative sentiment distribution chart to analyze trends across all IITs vs NITs.

- 🌟 **Library Experience Highlights** *(🌟 Library Experiences Tab)*  
  Shows handpicked positive and negative user reviews with institution tags and styled formatting.

---

### 💡 Future Improvements

- **Add LIME/SHAP Explainability for BERT**  
  Integrate model interpretation techniques to explain *why* a review was labeled positive/negative.

- **Include More Institutions**  
  Expand dataset to cover regional universities, IIITs, NLUs, and other public libraries for broader benchmarking.

- **Review Metadata Integration**  
  Include attributes like review date, source, device, or student/staff tag for richer context and filtering.

- **Clustering or Topic Modeling**  
  Apply LDA/BERT-topic to identify trending topics or issues discussed across institutions.

- **Sentiment Timeline Analysis**  
  Show how sentiment for a specific institution evolves over time (e.g., semester-wise or pre/post renovation).

- **User Feedback Module**  
  Allow users to correct or rate the model's prediction to improve performance and trust.

- **Multilingual Support**  
  Add language detection and support for Hindi, Tamil, etc., using multilingual BERT (e.g., `xlm-roberta-base`).

---

## 👨‍💻 Author
Aman Srivastava
[amansri345@gmail.com]

