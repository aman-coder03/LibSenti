# LibSenti: Library Review Sentiment Predictor & Analyst

LibSenti is an end-to-end AI-powered application that leverages machine learning and natural language processing (NLP) to classify sentiment polarity—Positive, Neutral, or Negative—from student-submitted reviews of IIT and NIT libraries. The system combines a robust backend sentiment analysis model with an interactive Streamlit application for real-time predictions, data visualization, and institutional insights.

This project focuses on handling class imbalance and improving neutral sentiment detection, which is a common challenge in real-world sentiment analysis systems.

---

## Key Features

- Real-time review prediction through an interactive interface  
- Visualization of sentiment probabilities for each prediction  
- Unigram word cloud generation for individual institutions  
- Bigram comparison between institutions for phrase-level insights  
- Sentiment distribution comparison using pie charts  
- IIT vs NIT aggregate sentiment analysis  
- Highlighted best and worst user experiences based on sentiment  

---

## Model Details

- Architecture: RoBERTaForSequenceClassification (roberta-base)  
- Dataset: IITs and NITs library reviews labeled as Positive, Neutral, or Negative  
- Frameworks: PyTorch, Hugging Face Transformers  
- Accuracy: ~95.3% (weighted F1-score optimized)
- Achieved balanced performance across all classes with minimal bias toward dominant classes

### Training Strategy

- Manual class weighting to address imbalance  
- Oversampling of the neutral class to improve recall  
- Label smoothing (0.1) for better generalization on ambiguous samples  
- Fine-tuned using Hugging Face Trainer with early stopping and best model selection based on weighted F1-score  

---

## Project Structure

LibSenti/
├── assets/
│   ├── wordclouds/  
│   └── iit_vs_nit_sentiment_comparison.png  
├── saved_model/  
├── app.py  
├── train_model.py  
├── cleaned_iit+nit_library_reviews.csv  
├── sentiment_iit_library_reviews.csv  
└── README.md  

---

## Sentiment Categories

| Label    | Class |
|----------|------|
| Negative | 0    |
| Neutral  | 1    |
| Positive | 2    |

Class imbalance is handled using a combination of:
- Manual class weighting in the loss function  
- Oversampling of the neutral class  
- Label smoothing to reduce overconfidence  

---

## Model Training

- Base Model: roberta-base  
- Framework: Hugging Face Transformers with PyTorch  

### Training Strategy

- Weighted cross-entropy loss for imbalance handling  
- Manually tuned class weights  
- Neutral class oversampling to improve F1-score  
- Label smoothing (0.1) for ambiguous sentiment handling  
- Maximum sequence length = 512 for improved context understanding  
- Learning rate of 8e-6 for stable fine-tuning  
- Early stopping to prevent overfitting  
- Best model selected based on weighted F1-score  

### Evaluation Metrics

- Accuracy  
- Weighted F1-score  
- Precision and Recall (per class)  

### Run Training

```bash
python train_model.py
```

### This Script Performs

- Data preprocessing  
- Tokenization  
- Model training  
- Class balancing  
- Model saving to `./saved_model/`  

---

## Performance Insights

- Strong performance across all classes (F1 ≈ 0.93–0.96)
- Neutral class performance significantly improved (F1 increased from ~0.82 to ~0.96) 
- Balanced performance achieved across all sentiment classes  

### Key Challenge

Neutral sentiment classification is inherently difficult due to semantic ambiguity.

### Solution Approach

- Data balancing  
- Label smoothing  
- Increased context window (512 tokens)  

---

## Streamlit Application

To launch the application:

```bash
streamlit run app.py
```

### Components

- Review classifier for real-time sentiment prediction  
- Probability visualization for model confidence  
- Word cloud comparison (unigram and bigram)  
- Sentiment distribution charts for institutions  
- IIT vs NIT comparative analysis  
- Highlighted user experiences  

---

## Future Improvements

- Integrate explainability tools such as LIME or SHAP  
- Expand dataset to include more institutions  
- Add metadata-based filtering (date, role, etc.)  
- Implement topic modeling for trend analysis  
- Introduce sentiment timeline visualization  
- Add user feedback loop for model improvement  
- Support multilingual sentiment analysis  

---

## Author

Aman Srivastava  
amansri345@gmail.com
