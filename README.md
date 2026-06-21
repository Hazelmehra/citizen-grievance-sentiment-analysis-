# AI-Driven Citizen Grievance & Sentiment Analysis System
An AI-powered Natural Language Processing system that automatically ingests citizen
grievances, classifies them into relevant complaint categories for efficient departmental
routing, and (in upcoming phases) performs sentiment analysis to prioritize issues based
on urgency and emotional tone.

This project was developed as part of a technical internship program focused on building
enterprise-grade, MLOps-aligned data science solutions for the Government & Public Sector
domain.

## Problem Statement
Public grievance redressal systems are often slow, disorganized, and reliant on manual
review. Civic officials must read through large volumes of citizen feedback to determine
relevance and urgency, resulting in delays and poor prioritization of critical issues.

This project addresses that gap by building an automated pipeline that:
- Cleans and analyzes raw citizen complaint text
- Classifies complaints into the correct category for routing
- (Upcoming) Scores complaints by sentiment and urgency to support prioritized response

## Dataset

**Source:** 311 Service Requests — New York City  
**Records used:** ~358,000 complaints (post-filtering)  
**Key columns used:**
| Column | Description |
|---|---|
| `Complaint Type` | Category of the complaint (e.g., Noise, Illegal Parking) |
| `Descriptor` | Detailed text description of the complaint |
| `Agency Name` | Government agency associated with the complaint |
| `Resolution Description` | Description of how the complaint was resolved |

## Project Roadmap

| Week | Focus | Status |
|---|---|---|
| Week 1 | Data Collection, Text Preprocessing & EDA | ✅ Complete |
| Week 2 | Topic Modeling & Complaint Categorization | ✅ Complete |
| Week 3 | Sentiment Analysis & Urgency Scoring | 🔄 In Progress |
| Week 4 | API Development & Deployment | ⬜ Pending |

## Week 1 — Data Preprocessing & Exploratory Data Analysis

**Objective:** Clean raw complaint text and explore patterns in citizen feedback.

**Key steps:**
- Loaded and filtered the 311 dataset to focus on well-represented complaint categories
- Built a complete text preprocessing pipeline:
  - Lowercasing
  - URL removal
  - Special character and punctuation removal
  - Tokenization
  - Stopword removal
  - Lemmatization
- Generated Word Cloud visualizations to identify frequently occurring complaint terms
- Performed Unigram and Bigram frequency distribution analysis
- Applied spaCy for Part-of-Speech tagging and Named Entity Recognition to explore
  grammatical structure and location references within complaint text
- Documented all findings with clear markdown explanations throughout the notebook

**Output:** `cleaned_complaints.csv`

## Week 2 — Topic Modeling & Complaint Categorization

**Objective:** Build a supervised model to automatically classify complaints by type.

**Key decision — Label Selection:**
During analysis, `Agency Name` was found to be heavily imbalanced (over 99% of records
belonged to a single agency), making it unsuitable as a classification target. `Complaint
Type` was selected instead, offering 13+ well-distributed categories suitable for
meaningful classification.

**Key decision — Avoiding Label Leakage:**
Input text was restricted to the `Descriptor` column only. Earlier combination of
`Complaint Type` with the input text was identified as a source of label leakage and
corrected.

**Modeling pipeline:**
- Text vectorized using **TF-IDF** (3,000 features)
- Class distribution proportions analyzed to surface minority-class risk
- Three models trained and compared:
  - **Naive Bayes** (baseline)
  - **Logistic Regression** (with `class_weight='balanced'`)
  - **Random Forest** (with `class_weight='balanced'`)
- Models evaluated using **Accuracy** and **Macro F1-score**, with Macro F1 prioritized
  to ensure fair performance across minority complaint categories
- **5-fold Cross-Validation** performed to confirm model generalization
- Confusion Matrix and full Classification Report generated for error analysis

**Results:**

| Model | Accuracy | Macro F1-Score |
|---|---|---|
| Naive Bayes (Baseline) | ~0.87 | ~0.74 |
| Logistic Regression | ~0.88 | ~0.78+ |
| Random Forest | ~0.88 | ~0.78+ |

**Final model selected:** Logistic Regression — chosen for its strong Macro F1
performance, fast inference time, and interpretability for deployment.

**Output:** `complaint_classifier_model.pkl`, `tfidf_vectorizer.pkl`

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy |
| NLP | NLTK, spaCy |
| Feature Engineering | Scikit-learn (TF-IDF) |
| Modeling | Scikit-learn (Naive Bayes, Logistic Regression, Random Forest) |
| Visualization | Matplotlib, Seaborn, WordCloud |
| Environment | Google Colab |

## Upcoming Work

- **Week 3:** Sentiment classification using a pretrained Transformer model
  (HuggingFace), with priority scoring based on detected urgency
- **Week 4:** Model serving via FastAPI, with a Streamlit interface for demonstration


