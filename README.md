# DocMind

A RAG-based research paper assistant built with LangChain, FAISS, HuggingFace embeddings, and Groq.

## What it does

Drop any PDF into `docmind/data/` and ask questions about it in plain English.
It retrieves the most relevant sections and answers using an LLM.

## How it works

1. Loads and chunks the PDF
2. Embeds chunks using sentence-transformers
3. Stores embeddings in a FAISS vector store
4. On each question — retrieves relevant chunks + always includes abstract
5. Passes context to Groq LLaMA for a grounded answer

## Stack

- LangChain — document loading and chunking
- HuggingFace sentence-transformers — embeddings
- FAISS — vector store
- Groq (LLaMA 3.1 8b) — LLM
