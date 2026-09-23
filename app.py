import streamlit as st
from transcript_parser import parse_transcript, parse_questions
from chunker import chunk_transcript
from vector_store import create_vector_store
from rag_chain import answer_question
from cross_analysis import analyze_cross_experts

# 1. Page Configuration
st.set_page_config(
    page_title="TranscriptIQ | Expert Transcript Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Initialize Session State
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "all_transcripts_data" not in st.session_state:
    st.session_state.all_transcripts_data = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "cross_results" not in st.session_state:
    st.session_state.cross_results = {}

# 3. App Header
st.title("🤖 TranscriptIQ")
st.caption("European Robotic Surgery Market Intelligence — Multi-Call Transcript Analyzer")
st.markdown("---")

# 4. File Paths
files = [
    "data/transcripts/Transcript_1_France.txt",
    "data/transcripts/Transcript_2_Germany.txt",
    "data/transcripts/Transcript_3_UK.txt"
]
interview_guide_path = "data/Interview_Guide.txt"

# 5. Sidebar Setup & Ingestion
with st.sidebar:
    st.header("⚙️ Data Management")
    if st.button("🚀 Ingest & Index Transcripts", type="primary", use_container_width=True):
        with st.spinner("Processing & indexing transcripts into FAISS..."):
            all_chunks = []
            for f in files:
                header, segments = parse_transcript(f)
                st.session_state.all_transcripts_data[header['name']] = (header, segments)
                chunks = chunk_transcript(header, segments)
                all_chunks.extend(chunks)

            st.session_state.vectorstore = create_vector_store(all_chunks)
            st.success("✅ 3 Transcripts Indexed Successfully!")

    st.markdown("---")
    if st.session_state.vectorstore is not None:
        st.success("🟢 System Ready")
        st.info(f"📚 {len(st.session_state.all_transcripts_data)} Experts Loaded")
    else:
        st.warning("⚠️ Please click the button above to index transcripts.")

    st.markdown("---")
    st.markdown("**Core Tech Stack:**")
    st.caption("• Embeddings: all-MiniLM-L6-v2")
    st.caption("• Vector Store: FAISS")
    st.caption("• LLM: Llama 3 / GPT-OSS on Groq")
    st.caption("• Anti-Hallucination: Grounded Citations")

# 6. Main 4 Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📄 Transcripts Viewer",
    "📋 Interview Guide",
    "🔍 Cross-Analysis",
    "💬 Ask Questions"
])

# ─── TAB 1: Transcripts Viewer ───
with tab1:
    st.header("📄 Transcripts Viewer")
    st.caption("Inspect parsed transcripts, speaker turns, and exact timestamps.")

    if not st.session_state.all_transcripts_data:
        st.info("👈 Please click **'Ingest & Index Transcripts'** in the sidebar to load the transcripts.")
    else:
        expert_options = list(st.session_state.all_transcripts_data.keys())
        selected_expert = st.selectbox("Select an Expert to Review:", expert_options)

        if selected_expert:
            header, segments = st.session_state.all_transcripts_data[selected_expert]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Expert", value=header.get("name"))
            with col2:
                st.metric(label="Role", value=header.get("role"))
            with col3:
                st.metric(label="Market", value=header.get("market"))

            st.markdown("---")
            st.subheader("Transcript Dialogue")

            for seg in segments:
                ts = seg.get("timestamp", "00:00")
                speaker = seg.get("speaker", "Speaker")
                text = seg.get("text", "")

                if "interviewer" in speaker.lower():
                    st.markdown(f"**🎤 {speaker}** `[{ts}]`")
                    st.write(text)
                else:
                    st.info(f"**💬 {speaker}** `[{ts}]`\n\n{text}")

# ─── TAB 2: Interview Guide Answers ───
with tab2:
    st.header("📋 Interview Guide Answers")
    st.caption("Automated answers to standardized questions, cited with exact quotes and timestamps.")

    if st.session_state.vectorstore is None:
        st.info("👈 Please index the transcripts first using the sidebar button.")
    else:
        questions = parse_questions(interview_guide_path)
        selected_q = st.selectbox("Select an Interview Guide Question:", questions)

        if selected_q:
            if st.button("Generate Expert Answers", type="primary"):
                with st.spinner("Retrieving answers per expert with exact citations..."):
                    experts = [
                        {"name": "Dr. Jean Martin", "market": "France", "flag": "🇫🇷"},
                        {"name": "Anna Keller", "market": "Germany", "flag": "🇩🇪"},
                        {"name": "Dr. Emily Carter", "market": "United Kingdom", "flag": "🇬🇧"}
                    ]
                    cols = st.columns(3)
                    for idx, exp in enumerate(experts):
                        with cols[idx]:
                            st.subheader(f"{exp['flag']} {exp['name']}")
                            st.caption(f"Market: {exp['market']}")
                            res = answer_question(selected_q, st.session_state.vectorstore, expert_name=exp["name"])
                            st.markdown(res["answer"])
                            with st.expander("📎 Retrieved Sources"):
                                for doc in res["sources"]:
                                    st.caption(f"**[{doc.metadata.get('timestamp')}]**: {doc.page_content}")

# ─── TAB 3: Cross-Transcript Analysis ───
with tab3:
    st.header("🔍 Cross-Transcript Analysis")
    st.caption("Synthesize common themes, contrasts, and disagreements across all 3 experts.")

    if st.session_state.vectorstore is None:
        st.info("👈 Please index the transcripts first using the sidebar button.")
    else:
        questions = parse_questions(interview_guide_path)
        selected_cross_q = st.selectbox("Select Question to Cross-Analyze:", questions, key="cross_q_select")

        if st.button("Run Cross-Analysis", type="primary"):
            with st.spinner("Analyzing common themes and disagreements across calls..."):
                res = analyze_cross_experts(selected_cross_q, st.session_state.vectorstore)
                st.session_state.cross_results[selected_cross_q] = res

        if selected_cross_q in st.session_state.cross_results:
            result_data = st.session_state.cross_results[selected_cross_q]
            st.markdown("### 📊 Cross-Analysis Synthesis")
            st.markdown(result_data["cross_analysis"])

            st.markdown("---")
            st.subheader("Individual Submissions")
            cols = st.columns(3)
            for idx, item in enumerate(result_data["individual_answers"]):
                with cols[idx]:
                    st.markdown(f"**{item['Expert']}**")
                    st.write(item["Answer"]["answer"])

# ─── TAB 4: Ask Questions (Interactive RAG Chat) ───
with tab4:
    st.header("💬 Ask Questions Across Transcripts")
    st.caption("Ask free-form questions — answers are strictly grounded with citations and timestamps.")

    if st.session_state.vectorstore is None:
        st.info("👈 Please index the transcripts first using the sidebar button.")
    else:
        # Display previous chat messages
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "sources" in msg and msg["sources"]:
                    with st.expander("📎 View Supporting Transcript Chunks"):
                        for s in msg["sources"]:
                            st.caption(f"**{s.metadata.get('expert_name')} ({s.metadata.get('timestamp')})**: {s.page_content}")

        # Chat Input Box
        if user_prompt := st.chat_input("Ask a question across all transcripts (e.g. 'How do budgets affect decisions?')..."):
            st.session_state.chat_history.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

            with st.chat_message("assistant"):
                with st.spinner("Searching across transcripts..."):
                    res = answer_question(user_prompt, st.session_state.vectorstore, expert_name=None)
                    st.markdown(res["answer"])
                    if res["sources"]:
                        with st.expander("📎 View Supporting Transcript Chunks"):
                            for s in res["sources"]:
                                st.caption(f"**{s.metadata.get('expert_name')} ({s.metadata.get('timestamp')})**: {s.page_content}")

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": res["answer"],
                "sources": res["sources"]
            })