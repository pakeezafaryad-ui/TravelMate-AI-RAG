# ✈️ TravelMate AI

TravelMate AI is a Retrieval-Augmented Generation (RAG) travel chatbot.

## Features

- 📄 Upload travel PDF documents
- 🔎 Semantic search using Hugging Face embeddings
- 🗄️ ChromaDB vector database
- 🤖 Groq LLM for answers
- 💬 Interactive Streamlit chatbot
- 📚 Shows the documents used as sources
- 🛡️ Reduces hallucinations by answering only from uploaded documents

## Technologies

- Python
- Streamlit
- Groq
- ChromaDB
- Hugging Face Sentence Transformers
- PyPDF
- LangChain Text Splitters

## How it works

PDF → Text Extraction → Chunking → Embeddings → ChromaDB → Retrieval → Groq → Answer

## Author

Pakeeza Faryad
