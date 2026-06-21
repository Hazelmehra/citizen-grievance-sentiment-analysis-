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


