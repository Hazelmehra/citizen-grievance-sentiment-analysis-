# Citizen Grievance and Sentiment Analysis System

An AI-powered Natural Language Processing (NLP) system that automatically ingests 
citizen complaints, classifies them into relevant complaint categories for efficient 
departmental routing, and performs sentiment analysis to prioritize issues based on 
urgency and emotional tone.

Developed as part of the Infotact Solutions Technical Internship Program — 
Data Science & Machine Learning track, Bengaluru.

## Problem Statement

Public grievance redressal systems are often slow, disorganized, and reliant on manual 
review. Civic officials must read through large volumes of citizen feedback to determine 
relevance and urgency, resulting in delays and poor prioritization of critical issues.

This project addresses that gap by building a fully automated dual-output NLP pipeline that:
- Cleans and analyzes raw citizen complaint text
- Classifies complaints into the correct category for departmental routing
- Scores complaints by sentiment and urgency for prioritized response
- Exposes both models via a FastAPI backend and Streamlit UI for real-world deployment

## Dataset

**Source:** 311 Service Requests — New York City  
**Total records:** ~358,000 complaints  
**Key columns used:**

| Column | Description |
|---|---|
| `Complaint Type` | Category of complaint (e.g., Noise, Illegal Parking) |
| `Descriptor` | Detailed text description of the complaint |
| `Agency Name` | Government agency associated with the complaint |
| `Resolution Description` | Description of how the complaint was resolved |

## Project Roadmap

| Week | Focus | Status |
|---|---|---|
| Week 1 | Data Collection, Text Preprocessing & EDA | ✅ Complete |
| Week 2 | Topic Modeling & Complaint Categorization | ✅ Complete |
| Week 3 | Sentiment Analysis & Urgency Scoring | ✅ Complete |
| Week 4 | API Development, Evaluation & Deployment | ✅ Complete |

## Week 1 — Data Preprocessing & Exploratory Data Analysis

**Objective:** Clean raw complaint text and explore patterns in citizen feedback.

**Key steps completed:**
- Loaded and filtered the 311 NYC dataset (~358,000 complaints)
- Built a complete text preprocessing pipeline:
  - Lowercasing
  - URL removal
  - Special character removal (replaced with space to avoid word merging)
  - Tokenization
  - Stopword removal
  - Lemmatization
- Generated Word Cloud visualizations for most frequent complaint terms
- Performed Unigram and Bigram frequency distribution analysis
- Applied spaCy for Part-of-Speech (POS) tagging and Named Entity Recognition (NER)
- Documented all findings with markdown explanations throughout the notebook

**Output:** `cleaned_complaints.csv`

## Week 2 — Topic Modeling & Complaint Categorization

**Objective:** Build a supervised model to automatically classify complaints by type.

**Key decisions:**

*Label Selection:* `Agency Name` was found to be heavily imbalanced (over 99% of records 
belonged to a single agency). `Complaint Type` was selected instead, providing 13+ 
well-distributed categories suitable for meaningful multi-class classification.

*Label Leakage Fix:* Input text was restricted to the `Descriptor` column only. Earlier 
combination of `Complaint Type` inside the input text was identified as label leakage 
and corrected before training.

**Modeling pipeline:**
- Text vectorized using TF-IDF (3,000 features)
- Class distribution proportions analyzed to identify minority-class risk
- Three models trained and compared:
  - Naive Bayes (baseline, as specified in MVP requirements)
  - Logistic Regression (with `class_weight='balanced'`)
  - Random Forest (with `class_weight='balanced'`)
- Models evaluated using Accuracy and Macro F1-score
- 5-fold Cross-Validation performed to confirm generalization
- Confusion Matrix and Classification Report generated

**Results:**

| Model | Accuracy | Macro F1-Score |
|---|---|---|
| Naive Bayes (Baseline) | 0.8736 | 0.7424 |
| Logistic Regression | 0.8751 | 0.7893 |
| Random Forest | 0.8751 | 0.7893 |

**Final model selected:** Logistic Regression — best Macro F1, simplest and most 
interpretable for deployment.

**Known limitation:** "Noise - Commercial" and "Noise - Street/Sidewalk" share nearly 
identical complaint text (both commonly use "loud music party"), making text-only 
classification unable to distinguish between them. Incorporating the `Location Type` 
column in future work could resolve this ambiguity.

**Output:** `complaint_classifier_model.pkl`, `tfidf_vectorizer.pkl`

## Week 3 — Sentiment Analysis & Urgency Scoring

**Objective:** Build a secondary model to classify the emotional tone and urgency of 
each complaint, enabling prioritization of critical issues.

**Approach:**
Since no human-labeled sentiment column exists in the dataset, sentiment labels were 
auto-generated using VADER (Valence Aware Dictionary and sEntiment Reasoner), a 
lexicon-based sentiment scorer. Labels were assigned across 4 classes:

| Class | Priority Score | Assignment Logic |
|---|---|---|
| Critical/Urgent | 4 | Urgency keywords (e.g., "emergency", "gas leak") OR compound score ≤ -0.3 |
| Negative | 3 | Compound score between -0.3 and -0.05 |
| Neutral | 2 | Compound score between -0.05 and +0.05 |
| Positive | 1 | Compound score ≥ +0.05 |

**Modeling pipeline:**
- Compound scores visualized to justify threshold selection
- Urgency keyword flagging applied for Critical/Urgent class
- TF-IDF vectorization on cleaned Descriptor text
- Two models trained and compared:
  - Logistic Regression (baseline)
  - LinearSVC (as recommended in project MVP specification)
- Best model selected based on Macro F1-score
- Confusion Matrix and Classification Report generated
- Numeric priority score (1–4) assigned to every complaint

**Output:** `sentiment_model.pkl`, `sentiment_vectorizer.pkl`

## Week 4 — API Development, Evaluation & Deployment

**Objective:** Package both trained models into a deployable application with a FastAPI 
backend and Streamlit frontend.

**Deliverables:**

*Final Evaluation:*
- Both models (department classifier + sentiment classifier) re-evaluated side by side
- Confusion Matrices and Classification Reports generated for final submission

*Inference Pipeline:*
- Single function accepts raw complaint text and returns:
  - Predicted complaint department/type
  - Predicted sentiment class
  - Numeric priority score (1–4)

*FastAPI Backend (`main.py`):*
- RESTful API accepting JSON payload with raw complaint text
- Returns predicted department and priority score as JSON response
- Endpoint: `POST /predict`

*Streamlit Frontend (`app.py`):*
- Simple UI for submitting complaint text and viewing predictions
- Displays department classification and urgency level visually
- Deployed via ngrok for public access from Google Colab

**Sample inference output:**
```json
{
  "department": "Illegal Parking",
  "sentiment": "Neutral",
  "priority_score": 2
}
```

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy |
| NLP | NLTK, spaCy, VADER |
| Feature Engineering | Scikit-learn (TF-IDF) |
| Modeling | Scikit-learn (Naive Bayes, Logistic Regression, Random Forest, LinearSVC) |
| Visualization | Matplotlib, Seaborn, WordCloud |
| API Framework | FastAPI, Uvicorn |
| Frontend | Streamlit |
| Deployment | ngrok (Colab tunnel) |
| Environment | Google Colab |

## Project Structure

```
citizen-grievance-sentiment-analysis/
│
├── notebooks/
│   └── Citizen_Grievance_Sentiment_Analysis_System.ipynb
│
├── main.py              ← FastAPI backend
├── app.py               ← Streamlit frontend
├── README.md
├── requirements.txt
└── .gitignore
```

> Trained model files (`.pkl`), datasets (`.csv`), and other large generated artifacts 
> are excluded from version control via `.gitignore`, in line with the project's 
> MLOps and data security guidelines.

## How to Run

1. Open the notebook in Google Colab
2. Upload the 311 NYC Service Requests dataset when prompted
3. Run all cells sequentially from top to bottom
4. Week 4 cells will automatically write `main.py` and `app.py`
5. Run the FastAPI cell to start the backend server
6. Run the Streamlit + ngrok cell to get a public URL for the UI

## Author

Developed as part of the AI/ML Technical Internship Program  
**Infotact Solutions**






