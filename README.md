# 🚀 Hybrid News Summarizer
### BART + Groq LLM | AI-Powered News Summarization System

A powerful **LLM-based news summarization platform** that scrapes news articles from the web, processes them using **Facebook BART**, and enhances summaries with **Groq-powered LLaMA models**.

The system supports **URL scraping, homepage crawling, batch summarization, text input summarization, and dataset evaluation**.

---

# 📌 Project Overview

Businesses and analysts often need to **quickly understand large volumes of news articles**.  
This project builds an **AI-powered summarization pipeline** that extracts key information from long news articles.

The system:

- Scrapes news articles from the web
- Processes them using **BART-large-CNN**
- Refines summaries using **Groq LLM**
- Extracts **key bullet points**
- Classifies news into categories
- Evaluates summaries using **ROUGE metrics**

---

# 🧠 Model Architecture

## Primary Model
**facebook/bart-large-cnn**

Why BART?

- Designed specifically for **abstractive summarization**
- Performs well on **news datasets**
- Strong contextual understanding

## Hybrid Enhancement

The BART output is optionally refined using **Groq LLaMA models** to improve:

- factual consistency
- readability
- coherence

Pipeline:

```
News Article
     ↓
Web Scraper
     ↓
BART Summarization
     ↓
Groq LLM Refinement
     ↓
Final Summary + Key Points + Classification
```

---

# ⚙️ System Architecture

```
User Input
   │
   ├── URL Scraping
   ├── Homepage Crawling
   ├── Batch Processing
   └── Direct Text Input
        │
        ▼
Scraping Engine (7 Layers)
        │
        ▼
Article Extraction
        │
        ▼
BART Summarization
        │
        ▼
Groq LLM Refinement
        │
        ▼
Results
 ├─ Summary
 ├─ Key Points
 ├─ Category
 └─ Evaluation Metrics
```

---

# 🌐 Advanced Web Scraping System

The application implements a **7-layer scraping fallback system**:

1. newspaper3k
2. requests + BeautifulSoup
3. meta-tag extraction
4. Google AMP mirror
5. archive.today cache
6. Wayback Machine snapshot
7. Freedium proxy for Medium articles

This increases **scraping success rate across different websites**.

---

# 🧩 Features

## 🌐 URL Article Summarization
Paste a news article URL and the system will generate:

- AI summary
- key bullet points
- news category
- compression statistics

---

## 🏠 Homepage Crawling

Paste a site homepage such as:

```
https://www.reuters.com
https://www.bbc.com
```

The system will:

1. detect homepage
2. crawl article links
3. allow user to select number of articles
4. summarize them automatically

---

## 📋 Batch Article Processing

Paste multiple URLs and process them simultaneously.

Output includes:

- summaries
- key insights
- article categories
- CSV export

---

## 📄 Direct Text Summarization

Paste article text directly to generate:

- summary
- bullet points
- classification

---

## 📰 XSum Dataset Demo

Includes demonstration using the **XSum news dataset**.

Allows comparison between:

- gold summaries
- AI generated summaries

---

## 📊 Model Evaluation

Example evaluation metrics:

| Metric | BART | Hybrid |
|------|------|------|
| ROUGE-2 | 0.212 | 0.235 |
| Fact Preservation | 72% | 91% |
| Number Accuracy | 68% | 94% |

---

# 🛠️ Tech Stack

## Core AI/ML

- PyTorch
- HuggingFace Transformers
- HuggingFace Datasets
- BART-large-CNN

## LLM Acceleration

- Groq API
- LLaMA models

## Web Scraping

- newspaper3k
- BeautifulSoup
- Requests
- Archive.today
- Wayback Machine

## App Framework

- Streamlit

## Data Processing

- Pandas
- NumPy

---

# 📂 Project Structure

```
Hybrid-News-Summarizer
│
├── app.py
├── requirements.txt
├── README.md
```

---

# 🚀 Installation

## 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/news-summarizer.git
cd news-summarizer
```

---

## 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate environment

Mac/Linux

```bash
source venv/bin/activate
```

Windows

```bash
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Run Application

```bash
streamlit run app.py
```

---

# 🔑 Groq API Setup (Optional)

Get a free API key:

```
https://console.groq.com
```

Enter the key in the **Streamlit sidebar** to enable hybrid summarization.

---

# 📊 Example Output

Example summary:

```
The U.S. Federal Reserve maintained interest rates while signalling
possible cuts later in the year as inflation continues to cool.
```

Example key points:

```
• Inflation slowed to 3.1%
• Federal Reserve kept rates unchanged
• Economic growth shows signs of slowing
• Markets expect rate cuts later this year
• Analysts warn about recession risks
```

---

# 🎯 Learning Objectives

This project demonstrates:

- LLM summarization pipelines
- hybrid AI architectures
- web scraping systems
- Streamlit AI application development
- ROUGE-based evaluation
- working with the XSum dataset

---

# 📌 Future Improvements

- Retrieval Augmented Generation (RAG)
- vector database integration
- multilingual summarization
- real-time news feeds
- GPU optimized inference

---

# 👨‍💻 Author

**Aviral Pratap Singh Chawda**  
AI / ML Engineer  
Gandhinagar, Gujarat, India

---

# ⭐ If you like this project

Give it a **star ⭐ on GitHub**
