# GenAI Project

Sales data analysis with LLM-powered SQL generation and document Q&A

Overview

Smart Insights Hub is a multi-agent AI system that supports:

Natural language → SQL query generation

SQL validation

Natural language response generation

For realtime uploaded files :

  -Smart summarization from DataFrames
  -RAG-based summarization for PDF & Word documents

Secure user login with hashed credentials

Modular Streamlit UI with tabs

## Features
- SQL Q&A with LangChain agents
- Document summarization and analysis
- Real-time data insights
- SQL Query Agent (Groq LLM)
- SQL Validation Agent
- NL Summary Agent
- Document RAG Agent
- Supports PDF, DOCX
- DuckDB storage
- Secure bcrypt user login system

## Installation
```
pip install -r requirements.txt
```

## Running the App
```
streamlit run app.py
```
