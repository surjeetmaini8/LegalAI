# Legal AI Chat Assistant 🤖

An advanced RAG-based intelligent chatbot for legal document analysis, Q&A, summarization, and comparison.

## 🌟 Features

- **Legal Q&A**: Ask questions and get context-grounded answers from legal documents
- **Document Summarization**: Summarize long judgments or contracts into concise points
- **Clause Comparison**: Compare two legal documents and highlight semantic differences
- **Voice Mode**: Speak your queries and hear responses

## 🛠️ Tech Stack 

- **LLM**: Google Flan-T5 (free, runs locally)
- **RAG Framework**: LangChain
- **Vector Store**: FAISS
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **Summarization**: BART-Large-CNN
- **Speech-to-Text**: OpenAI Whisper (free, runs locally)
- **Text-to-Speech**: gTTS
- **UI**: Streamlit

## 📦 Installation

1. **Clone and setup**:
```bash
cd legal
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

2. **Create data directories**:
```bash
mkdir data
mkdir data\documents
mkdir data\vector_store
```

3. **Add your legal documents**:
Place PDF/DOCX files in `data/documents/`

4. **Run the app**:
```bash
streamlit run app.py
```

## 📚 Usage

### 1. Q&A Mode
- Ask: "What is the punishment for criminal breach of trust under IPC?"
- The system retrieves relevant passages and provides contextual answers

### 2. Summarization
- Upload a judgment or select from existing documents
- Get 5-point summary with key facts, decision, legal basis, precedent, and verdict

### 3. Clause Comparison
- Upload two contracts or documents
- Get semantic differences and legal implications

### 4. Voice Mode
- Click microphone icon
- Speak your query
- Listen to the answer

## 🗂️ Project Structure

```
legal/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Dependencies
├── .env.example               # Environment variables template
├── config.py                  # Configuration management
├── modules/
│   ├── document_processor.py  # Document loading and chunking
│   ├── vector_store.py        # FAISS vector store management
│   ├── rag_engine.py          # RAG Q&A system
│   ├── summarizer.py          # Document summarization
│   ├── comparator.py          # Clause comparison
│   └── voice_handler.py       # Speech-to-text & text-to-speech
├── data/
│   ├── documents/             # Your legal documents (PDF/DOCX)
│   └── vector_store/          # FAISS index files
└── utils/
    └── helpers.py             # Utility functions
```

## 🚀 Performance Tips

- First run downloads models (~2GB total)
- For faster inference, use smaller models in config
- GPU recommended but not required

## 📄 License

MIT License - Free for personal and commercial use
