# 🤖 TranscriptIQ

**European Robotic Surgery Market Intelligence — Multi-Call Transcript Analyzer**

TranscriptIQ is a Retrieval-Augmented Generation (RAG) application that parses expert-interview transcripts, indexes them into a vector store, and lets you query, compare, and cross-analyze what multiple experts said — with every answer grounded in exact quotes and timestamps.

---

## ✨ Features

- **Transcript Viewer** — inspect parsed transcripts with speaker turns and exact timestamps
- **Interview Guide Answers** — get each expert's answer to a standardized question, side by side
- **Cross-Transcript Analysis** — synthesize common themes and disagreements across all experts for a given question
- **Ask Questions (Chat)** — free-form Q&A across all transcripts, with citations
- **Grounded citations** — every answer cites `[Expert Name, Market, Timestamp: MM:SS]`, and the model is instructed to say *"Information not found in the transcripts"* rather than guess

---

## 🧱 Tech Stack

| Component | Choice |
|---|---|
| UI | Streamlit |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| Vector Store | FAISS (via LangChain) |
| LLM | `openai/gpt-oss-120b` on Groq |
| Orchestration | LangChain (prompts, chains, output parsing) |

---

## 📂 Project Structure

```
TranscriptIQ/
├── app.py                  # Streamlit UI — 4 tabs (Viewer, Guide, Cross-Analysis, Chat)
├── transcript_parser.py    # Parses .txt transcripts and the interview guide
├── chunker.py               # Pairs interviewer Q + expert A into retrieval chunks
├── vector_store.py         # Builds the FAISS index; retrieves + filters by expert
├── prompts.py               # LangChain prompt templates (answer, cross-analysis, chat)
├── rag_chain.py             # Core RAG pipeline — retrieve, format context, call LLM
├── cross_analysis.py        # Gathers per-expert answers and synthesizes across them
├── data/
│   ├── transcripts/
│   │   ├── Transcript_1_France.txt
│   │   ├── Transcript_2_Germany.txt
│   │   └── Transcript_3_UK.txt
│   └── Interview_Guide.txt
└── .env                     # GROQ_API_KEY (not committed)
```

---

## ⚙️ Setup

**1. Clone and install dependencies**

```bash
pip install streamlit langchain langchain-community langchain-huggingface langchain-groq faiss-cpu sentence-transformers python-dotenv
```

**2. Add your Groq API key**

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

**3. Add transcript data**

Place your transcript `.txt` files under `data/transcripts/` and the interview guide at `data/Interview_Guide.txt`, following the expected format:

```
Expert 1 – Dr. Jean Martin
Role: Head of Urology
Market: France

00:00
Interviewer: <question>

00:18
Dr. Martin: <answer>
```

**4. Run the app**

```bash
streamlit run app.py
```

---

## 🧠 How It Works

1. **Parse** — `transcript_parser.py` extracts the header (name, role, market) and a list of timestamped speaker segments from each transcript.
2. **Chunk** — `chunker.py` pairs each Interviewer question with the following expert answer into one retrieval unit, tagged with expert metadata and the answer's timestamp.
3. **Embed & Index** — `vector_store.py` embeds each chunk with `all-MiniLM-L6-v2` and stores it in a FAISS index via LangChain, keeping the original text and metadata attached to each vector.
4. **Retrieve & Answer** — `rag_chain.py` retrieves the most relevant chunks for a question (optionally filtered to one expert), formats them into a structured context block, and asks the LLM to answer strictly from that context with citations.
5. **Cross-Analyze** — `cross_analysis.py` runs the retrieval-and-answer step once per expert, then asks the LLM to synthesize agreements and disagreements across all of them.

## 🎯 Key Design Decisions & Scaling

### 1. Hallucination Reduction Strategy
- **Strict Prompt Grounding**: System prompts explicitly enforce: *"Rely ONLY on the provided context. If the context does not contain the answer, output 'Information not found in the transcripts.'"*
- **Verifiable Direct Quotes**: All key claims are instructed to be wrapped in exact quotation marks (`"..."`).
- **Temperature = 0.0**: Eliminates random generation and forces deterministic, fact-bound responses.
- **UI Traceability**: Every answer includes an expandable **"📎 View Supporting Transcript Chunks"** dropdown showing the exact raw retrieved text and timestamp for full auditability.

### 2. Handling Citations & Timestamps
- Rather than arbitrary character splitting, transcripts are chunked at **natural speaker-turn boundaries**.
- The expert's timestamp (`MM:SS`), speaker name, role, and market are embedded directly into each LangChain `Document` metadata:
```python
  metadata = {"expert_name": "Dr. Jean Martin", "market": "France", "timestamp": "01:20"}
```
- Retrieved chunks inject these tags into the prompt context so the model cites them in standard format: [Expert Name, Market, Timestamp: MM:SS]

### 3. Scaling from 3 to 30+ Transcripts
- **Managed Vector Database**: Currently, FAISS runs in-memory, which is extremely fast for a small number of transcripts. For 30+ or hundreds of transcripts, we would persist the vector index to disk or deploy a centralized vector database so the index doesn't have to be rebuilt every session.
- **Metadata Pre-Filtering**: In our current retrieval, we filter by expert after retrieving chunks. With 30+ experts, we Apply database-level filters (filtering by market, specialty, or date) prior to calculating vector similarity to keep search fast and accurate.
- **Batch Processing**: Instead of parsing files one by one on startup, we would process new transcripts in batches and update the vector store incrementally.

## 👤 Author
**Ramya Sri Immadi**  
*ML / AI Engineer*