🩺 MediBot — RAG-Powered Medical Chatbot   

Live Demo: https://rag-medical-chatbot-fqpgbpjmszsi4hkuu42q7p.streamlit.app/

MediBot is a Retrieval-Augmented Generation (RAG) chatbot that answers medical questions by reading and reasoning over your own medical PDF documents instead of relying purely on what a language model "remembers" from training. It retrieves the most relevant passages from a local knowledge base and uses a large language model to turn those passages into a clear, detailed answer.


Table of Contents

Why I Built This
What is RAG (and why use it here)
How MediBot Works
Tech Stack
Project Structure
Pipeline Walkthrough
Setup & Installation Guide
Usage
Example
Limitations & Disclaimer
Future Improvements
License



Why I Built This

General-purpose chatbots like a base LLM are great conversationalists, but they have two big problems when it comes to medical information:


They hallucinate. If they don't know an answer, they often make one up confidently instead of saying "I don't know."
They don't know your specific sources. A doctor, student, or researcher usually wants answers grounded in a specific trusted reference (a textbook, a clinical manual, a research PDF) not a vague, generic answer from the entire internet.


I built MediBot to solve this with Retrieval-Augmented Generation (RAG): instead of asking the LLM to answer from memory, I first retrieve the most relevant chunks of text from a curated set of medical PDFs, and then force the LLM to answer only using that retrieved context. If the answer isn't in the documents, the bot is instructed to say "I don't know" rather than invent information.

This project was also a way to learn and demonstrate the full RAG pipeline end-to-end — document ingestion, chunking, embeddings, vector search, prompt engineering, and a usable chat UI using free/open-source tools wherever possible.


What is RAG (and why use it here)

Retrieval-Augmented Generation combines two things:


Retrieval: A search step that finds the most relevant pieces of text from a knowledge base (here, medical PDFs) based on the user's question.
Generation: An LLM that reads the retrieved text and generates a natural-language answer grounded in it.


Instead of fine-tuning a model (expensive, slow, and quickly outdated), RAG lets you swap or update the knowledge base any time just add new PDFs and rebuild the vector index. The LLM itself never needs to be retrained.

For a medical use case specifically, this matters because:


Answers can be traced back to source documents (source_documents is returned alongside every answer).
The model is explicitly instructed not to fabricate facts outside the given context.
You control exactly which medical references the bot is allowed to "know."



How MediBot Works

At a high level, the project has two phases:

Phase 1 — Build the Knowledge Base (offline, run once / whenever sources change)

Load all PDF files from the data/ folder.
Merge multi-page PDFs into a single document per file (so context isn't broken mid-page).
Split the merged text into overlapping chunks.
Convert each chunk into a vector embedding using a sentence-transformer model.
Store all embeddings in a local FAISS vector index on disk.


Phase 2 — Ask Questions (runtime)


The user asks a question (via terminal or the Streamlit web app).
The question is embedded using the same embedding model.
FAISS performs a similarity search and returns the top-k most relevant chunks.
Those chunks are inserted into a custom prompt template as context.
The prompt + context + question are sent to a Groq-hosted LLaMA 3.3 70B model.
The model generates an answer using only the provided context, and the matching source documents are shown alongside it.


PDFs (data/) 
   │
   ▼
Load & Merge Pages  (create_memory_for_llm.py)
   │
   ▼
Chunking (1000 chars, 300 overlap)
   │
   ▼
Embeddings (all-MiniLM-L6-v2)
   │
   ▼
FAISS Vector Store (vectorstore/db_faiss)
   │
   ▼
User Question ──► Retriever (top-k=10) ──► Context
                                              │
                                              ▼
                                    Prompt Template + LLM (Groq LLaMA 3.3 70B)
                                              │
                                              ▼
                                    Answer + Source Documents
                                    (connect_memory_with_llm.py — CLI
                                     or Medibot.py — Streamlit UI)


Tech Stack

ComponentTool / LibraryPurposeOrchestrationLangChain (langchain, langchain-community, langchain-core)Glue layer connecting loaders, splitters, embeddings, retriever, and LLMPDF LoadingPyMuPDFLoader + DirectoryLoaderReads all PDFs from the data/ folderText SplittingRecursiveCharacterTextSplitterBreaks documents into overlapping chunks for embeddingEmbeddingslangchain-huggingface → sentence-transformers/all-MiniLM-L6-v2Converts text chunks into vector representations (free, local, no API key needed)Vector DatabaseFAISS (Facebook AI Similarity Search)Stores and searches embeddings locally — no external DB requiredLLMlangchain-groq → LLaMA 3.3 70B Versatile via Groq APIGenerates the final answer (fast inference, free tier available)Web UIStreamlitSimple chat interface (Medibot.py)Configpython-dotenvLoads API keys from a .env fileLanguagePython 100%Entire project is pure Python


Project Structure

RAG-Medical-Chatbot/
│
├── data/                       # Put your source medical PDFs here
├── vectorstore/
│   └── db_faiss/                # Generated FAISS index (created automatically)
├── create_memory_for_llm.py     # Step 1: build the vector store from PDFs
├── connect_memory_with_llm.py   # Step 2: command-line Q&A using the vector store + LLM
├── Medibot.py                   # Step 3: Streamlit chat web app
├── .gitignore
└── .env                         # (you create this) — holds your API keys, not committed


Note: the original repo doesn't include a requirements.txt. The exact packages needed are listed in Setup & Installation below — you can paste them into a requirements.txt of your own.



Pipeline Walkthrough

1. create_memory_for_llm.py — Build the Knowledge Base


Loads every .pdf in data/ with PyMuPDFLoader.
Merges all pages of each source file into a single Document, sorted by page number, so a chunk never gets awkwardly cut off mid-page boundary metadata.
Splits the merged text using RecursiveCharacterTextSplitter with chunk_size=1000 and chunk_overlap=300 (overlap helps preserve context across chunk boundaries).
Embeds chunks with the all-MiniLM-L6-v2 sentence-transformer model (384-dimension embeddings, runs locally on CPU, no API cost).
Saves everything into a local FAISS index at vectorstore/db_faiss.


2. connect_memory_with_llm.py — Command-Line Testing


Loads the saved FAISS index.
Defines a custom prompt template that instructs the model to:

Answer in 4–6 clear sentences or bullet points.
Include clinical details (causes, symptoms, treatment) if present in the context.
Say "I don't know" if the answer isn't in the context — never fabricate.
Use only the given context, nothing outside it.



Builds a RetrievalQA chain (chain_type="stuff") with the retriever set to return the top 10 most relevant chunks (k=10).
Takes a question from the terminal, prints the answer and the source documents used.


3. Medibot.py — Streamlit Chat App


Same retrieval + prompt logic as above, wrapped in a Streamlit chat interface.
Maintains conversation history in st.session_state.
Caches the vector store load with @st.cache_resource so it isn't reloaded on every interaction.
Displays the model's answer along with the retrieved source documents directly in the chat bubble.



Setup & Installation Guide

Follow these steps exactly to get MediBot running on your own machine.

Step 1 — Clone the repository

bashgit clone https://github.com/sagramanisahil/RAG-Medical-Chatbot.git
cd RAG-Medical-Chatbot

Step 2 — Create a virtual environment (recommended)

bashpython -m venv venv

# Activate it:
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

Step 3 — Install dependencies

This repo doesn't ship a requirements.txt, so install the packages used directly:

bashpip install langchain langchain-community langchain-core langchain-huggingface langchain-groq
pip install streamlit
pip install pymupdf
pip install faiss-cpu
pip install sentence-transformers
pip install python-dotenv

Or save this as requirements.txt and run pip install -r requirements.txt:

langchain
langchain-community
langchain-core
langchain-huggingface
langchain-groq
streamlit
pymupdf
faiss-cpu
sentence-transformers
python-dotenv

Step 4 — Get a free Groq API key


Go to console.groq.com and sign up (free tier available).
Create an API key.


Step 5 — Create your .env file

In the project root, create a file named .env:

GROQ_API_KEY=your_groq_api_key_here

Step 6 — Add your medical PDFs

Place any medical reference PDFs (textbooks, manuals, clinical guides, etc.) you want the bot to learn from inside the data/ folder:

data/
├── medical_book_1.pdf
├── medical_book_2.pdf


⚠️ Only use PDFs you have the legal right to use. Don't upload copyrighted material you don't own or have permission to use.



Step 7 — Build the vector store

Run this once (and again any time you add/change PDFs):

bashpython create_memory_for_llm.py

This reads all PDFs in data/, chunks them, embeds them, and saves the FAISS index to vectorstore/db_faiss/.

Step 8 — Test it from the command line (optional)

bashpython connect_memory_with_llm.py

You'll be prompted: Write Query Here: — type a medical question and press enter.

Step 9 — Launch the chatbot web app

bashstreamlit run Medibot.py

This opens a browser tab with the MediBot chat interface at http://localhost:8501.


💬 Usage


Make sure vectorstore/db_faiss/ exists (run Step 7 above first if not).
Run streamlit run Medibot.py.
Type a question related to the content of your PDFs into the chat box, e.g.:

"What are the symptoms of Type 2 diabetes?"
"What is the recommended treatment for hypertension?"



MediBot retrieves the most relevant chunks from your PDFs and generates an answer, along with the source document chunks it used.



Example

You: What causes anemia?

MediBot:
Anemia is primarily caused by a deficiency in red blood cells or hemoglobin...
- Iron deficiency due to poor diet or blood loss
- Vitamin B12 or folate deficiency
- Chronic diseases affecting red blood cell production
- Genetic conditions such as sickle cell anemia

Source Docs: [chunks retrieved from your PDF, with file name and page metadata]


Limitations & Disclaimer


This is not a medical device and must not be used for real diagnosis or treatment decisions. It's an educational/learning project demonstrating RAG architecture.
Answer quality depends entirely on the PDFs you provide — garbage in, garbage out.
The retriever returns the top 10 chunks by similarity; it can occasionally miss the most relevant passage if your documents are very large or poorly structured.
The allow_dangerous_deserialization=True flag is required to load FAISS pickled indexes — only load vector stores you created yourself, not ones from untrusted sources.
No authentication, rate limiting, or PII handling is implemented — don't deploy this publicly with sensitive data without adding proper security.



Future Improvements


Add a requirements.txt / pyproject.toml to the repo for one-command installs.
Add citation formatting in the UI (clickable source page numbers instead of raw dumps).
Support more document types (Word, plain text, web pages).
Add conversational memory so follow-up questions retain context.
Add evaluation metrics (retrieval precision/recall, answer faithfulness).
Containerize with Docker for easier deployment.



License

No license file is currently specified in the repository. Add a LICENSE file (e.g., MIT) if you intend others to reuse this code.


🙌 Credits

Built by sagramanisahil using LangChain, FAISS, HuggingFace Sentence Transformers, Groq, and Streamlit.
