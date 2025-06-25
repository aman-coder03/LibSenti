# 📚 LibSenti - Library Review Sentiment Analysis Dashboard

LibSenti is an end-to-end NLP application that uses a fine-tuned BERT model to predict sentiments (Positive, Neutral, Negative) from user-submitted library reviews of IITs and NITs. It also features an interactive Streamlit-based dashboard for sentiment prediction, wordcloud exploration per institution, and a comparative sentiment analysis chart between IITs and NITs.

---

## 🚀 Features

- ✅ **Sentiment Prediction** using a BERT model fine-tuned on IIT/NIT library reviews
- 🎯 **Adjusted thresholding** for more reliable sentiment classification
- ☁️ **Institution-specific WordClouds** dynamically loaded from disk
- 📊 **Sentiment Comparison Chart** for IITs vs NITs (static image)
- 🖼️ **Streamlit Dashboard** with clean UI and prediction feedback

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

## 🎨 Streamlit Dashboard
Launch the dashboard:
```
streamlit run app.py
```
Components:
- 📥 Review Text Box: Type or paste a review and classify it
- 📈 Sentiment Probabilities: Visual breakdown using progress bars
- 📌 WordCloud Selector: Choose an IIT/NIT to view wordcloud
- 📊 IIT vs NIT Chart: Static PNG bar chart

---

## 💡 Future Improvements
- Add LIME/SHAP explanations for BERT predictions
- Replace static bar chart with dynamic matplotlib/plotly chart
- Include more regional/NLUs/IIITs for broader analysis
- Add review source metadata or clustering
