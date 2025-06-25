# 📚 LibSenti: Library Review Sentiment Predictor & Analyst

LibSenti is an end-to-end **AI/ML and NLP-driven application** that predicts sentiment polarity (Positive, Neutral, Negative) from user-submitted library reviews of IITs and NITs. It uses a fine-tuned **BERT-based transformer model** for classification and provides interactive insights through a visually rich **Streamlit dashboard**.

## 🚀 Key Features

- ✅ **Sentiment Classification using BERT**  
  Fine-tuned BERT transformer model (3-class classification) to predict review sentiment with high accuracy.

- ✅ **Real-time Review Prediction**  
  Users can input custom reviews to receive live sentiment prediction with confidence probabilities.

- ✅ **Real-Time WordCloud Visualization**  
  Dynamically displays institution-specific wordclouds based on selected reviews, enabling term frequency exploration.

- ✅ **Sentiment Distribution Pie Chart Comparison**  
  Provides real-time sentiment pie chart comparison between any two institutions, allowing intuitive visual analysis of Positive, Neutral, and Negative sentiments.

- ✅ **Dual Institution Analysis**  
  Compare **two institutions simultaneously** using:
  - 📊 WordClouds  
  - 📈 Sentiment Pie Charts (with leader lines and percentage labels)

- ✅ **IIT vs NIT Overall Sentiment Distribution**  
  A static comparison chart showing aggregated sentiment trends across all IITs and NITs.

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

- 📥 **Review Classifier**  
  Enter any library review and instantly receive a sentiment prediction (Positive, Neutral, Negative).

- 📈 **Sentiment Probabilities**  
  Visualize the confidence scores for each sentiment using interactive progress bars.

- ☁️ **WordCloud Comparator**  
  Select and compare two institutions to view their real-time wordclouds based on review text analysis.

- 📊 **Sentiment Pie Chart Comparison**  
  Instantly loads sentiment distribution charts for selected institutions side-by-side for intuitive visual analysis.

- 🧮 **IIT vs NIT Overall Chart**  
  A comparative sentiment distribution chart to analyze trends across all IITs vs NITs.

---

## 💡 Future Improvements
- Add LIME/SHAP explanations for BERT predictions  
- Include more regional/NLUs/IIITs for broader analysis
- Add review source metadata or clustering

---

## 👨‍💻 Author
Aman Srivastava
[amansri345@gmail.com]

