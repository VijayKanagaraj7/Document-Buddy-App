# 📄 Document Buddy App
![WhatsApp Image 2025-07-20 at 16 23 27_67ade186](https://github.com/user-attachments/assets/de52147a-c1ce-49f6-a764-30d3596494de)
![WhatsApp Image 2025-07-20 at 16 23 27_8e0afd0c](https://github.com/user-attachments/assets/0dae3841-8195-4817-927b-9580e2bc5270)
![WhatsApp Image 2025-07-20 at 16 23 27_6dcfa6ad](https://github.com/user-attachments/assets/1de323e0-647d-4ef4-97e0-5b266b5d52eb)

**Your Personal Document Assistant powered by Generative AI, Embeddings & ChromaDB**

![badge](https://img.shields.io/badge/LLM-Gemini--2.5--Flash-blue) ![badge](https://img.shields.io/badge/Embeddings-BAAI%2Fbge--small--en-green) ![badge](https://img.shields.io/badge/Framework-LangChain-orange) ![badge](https://img.shields.io/badge/Frontend-Streamlit-red) ![badge](https://img.shields.io/badge/DB-Chroma-purple)

---

## 🔍 Overview

**Document Buddy App** enables you to interact with PDF documents using Google Gemini LLM via Retrieval-Augmented Generation (RAG). It allows:
- 🧠 Semantic search over your PDFs.
- 💬 Intelligent Q&A from the uploaded document.
- ⚙️ Embedding generation via `BAAI/bge-small-en` model.
- 🗂️ Vector storage with ChromaDB.
- 🎨 Clean, interactive UI using Streamlit.

> Ideal for researchers, professionals, and developers looking to enhance productivity with AI-powered document understanding.

---

## 🚀 Features

- 📂 Upload and parse PDFs
- 🔄 Auto-generate vector embeddings (HuggingFace BGE model)
- 🧠 RAG-powered chatbot using Google Gemini API
- 💬 Interactive Streamlit chat interface
- 💾 Persistent local vector database using ChromaDB
- 🔒 Error handling for document and model issues

---

## 🛠️ Tech Stack

| Component       | Technology                             |
|----------------|-----------------------------------------|
| LLM            | Google Generative AI (Gemini 2.5 Flash) |
| Embeddings     | HuggingFace BGE (`bge-small-en`)        |
| Vector DB      | ChromaDB                                |
| Framework      | LangChain                               |
| UI             | Streamlit                               |
| Backend Lang   | Python 3.x                              |

---

## 📸 UI Preview

| Upload PDF | Chatbot |
|------------|---------|
| ![Upload UI](https://via.placeholder.com/300x200?text=Upload+PDF+UI) | ![Chatbot UI](https://via.placeholder.com/300x200?text=Chatbot+UI) |

---

## 📂 Project Structure
📁 document-buddy-app/
├── app.py # Streamlit frontend and routing
├── chatbot.py # Chatbot manager using LangChain and Gemini
├── vectors.py # PDF loader, splitter, and Chroma embedding manager
├── db/ # ChromaDB vector storage (auto-created)
├── temp.pdf # Temporary uploaded file (auto-generated)
└── README.md # Project documentation

---

## ⚙️ Setup & Installation

### 1. 🧬 Clone the Repo
```bash
git clone https://github.com/AIAnytime/Document-Buddy-App.git
cd Document-Buddy-App

2. 📦 Install Dependencies
Make sure you have Python 3.8+
pip install -r requirements.txt

3. 🔑 Set Google Gemini API Key
Create a .env file or export it directly:
export GOOGLE_API_KEY="your_gemini_api_key_here"

4. 🚀 Run the App
streamlit run app.py

🧠 How It Works
PDF Upload: Load and parse PDFs using PyPDFLoader.

Text Splitting: Documents are chunked using RecursiveCharacterTextSplitter.

Embedding Generation: Each chunk is embedded using BAAI/bge-small-en.

Vector Storage: Embeddings are persisted using ChromaDB.

Retrieval: Top-K relevant chunks are retrieved.

Answer Generation: Gemini-2.5-Flash LLM generates a context-aware answer.

UI Display: Streamlit displays the full conversational flow.

🔐 Security Notes
API Key is currently hardcoded in chatbot.py. Use environment variables or python-dotenv for secure usage.

Uploaded files are saved locally as temp.pdf. You can add cleanup logic or use a temp directory.

✅ To-Do / Enhancements
 Add support for multiple file types (DOCX, TXT)

 Add citation support from source chunks

 Support document summaries

 GPU support for faster embeddings (via CUDA)

 Authentication & session-based storage

🤝 Contributing
We welcome contributions from the community!

Fork this repo 🍴

Create your feature branch (git checkout -b feature-name)

Commit your changes (git commit -m 'add feature')

Push to the branch (git push origin feature-name)

Open a Pull Request ✅

📬 Contact
Email: vijaykanagaraj1986@gmail.com
GitHub: AIAnytime/Document-Buddy-App

🌟 Acknowledgements
LangChain
Google Generative AI
HuggingFace
ChromaDB
Streamlit
