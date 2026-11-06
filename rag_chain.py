from langchain_groq import ChatGroq
from typing import List
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder
from vector_store import retrive_chunks
import os
from dotenv import load_dotenv
load_dotenv()
from prompts import ANSWER_QUESTION_PROMPT, EVIDENCE_SUFFICIENCY_PROMPT,CHAT_PROMPT

model = ChatGroq(
    temperature=0.0,
    model_name="openai/gpt-oss-120b",
    groq_api_key=os.environ.get("GROQ_API_KEY")
    )

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

def format_transcript_context(documents: List[Document]) -> str :
    """
    Takes a list of retrieved LangChain Document objects and converts them into
    a clean, readable string formatted with strict metadata headers for the LLM.
    """
    formatted_chunks = []
    
    for i, doc in enumerate(documents, start=1):
        # Extract metadata fields with fallback values if they are missing
        expert = doc.metadata.get("expert_name", "Unknown Expert")
        market = doc.metadata.get("market", "Unknown Market")
        timestamp = doc.metadata.get("timestamp", "00:00")
        
        # Clean up the text chunk (remove trailing white spaces/newlines)
        text_content = doc.page_content.strip()
        
        # Build the structured block for this specific document chunk
        chunk_str = (
            f"[Transcript Chunk {i}]\n"
            f"Expert Name: {expert}\n"
            f"Market: {market}\n"
            f"Timestamp: {timestamp}\n"
            f"Text: {text_content}"
        )
        formatted_chunks.append(chunk_str)
        
    # Join all chunk blocks together with double newlines for clear separation
    return "\n\n".join(formatted_chunks)

def rerank_chunks(question, documents, top_k=5):
    """
    Reranks retrieved transcript chunks using a Cross-Encoder.
    Returns the top_k most relevant chunks.
    """

    if not documents:
        return []

    pairs = [
        [question, doc.page_content]
        for doc in documents
    ]

    scores = reranker.predict(pairs)

    scored_documents = list(zip(documents, scores))

    ranked_documents = sorted(
        scored_documents,
        key=lambda item: item[1],
        reverse=True
    )

    return [
        doc
        for doc, score in ranked_documents[:top_k]
    ]

def answer_question(question, vectorstore, expert_name=None):

    # Step 1: Retrieve 20 candidate chunks
    candidate_chunks = retrive_chunks(question, vectorstore, expert_name=expert_name,k=20)

    # Step 2: Rerank and select the top 5
    retrieved_chunks = rerank_chunks(
        question,
        candidate_chunks,
        top_k=5
    )

    if not retrieved_chunks:
        return {
            "answer": "Information not found in the transcripts.",
            "sources": [],
            "evidence_status": "INSUFFICIENT"
        }

    # Step 3: Check evidence sufficiency
    evidence_status = check_evidence_sufficiency(
        question,
        retrieved_chunks
    )

    if evidence_status != "SUFFICIENT":
        return {
            "answer": "Information not found in the transcripts.",
            "sources": [],
            "evidence_status": evidence_status
        }

    # Step 4: Format the selected chunks
    formatted_chunks = format_transcript_context(retrieved_chunks)

    # Step 5: Generate the grounded answer
    chain = ANSWER_QUESTION_PROMPT | model | StrOutputParser()

    response = chain.invoke({
        "context": formatted_chunks,
        "question": question
    })

    return {
        "answer": response,
        "sources": retrieved_chunks,
        "evidence_status": evidence_status
    }


def check_evidence_sufficiency(question, documents):
    context = format_transcript_context(documents)

    prompt = EVIDENCE_SUFFICIENCY_PROMPT.format(
        question=question,
        context=context
    )

    response = model.invoke(prompt)

    return response.content.strip().upper()

def answer_chat_question(question, vectorstore, chat_history=None):

    chat_history = chat_history or []

    # Step 1: Build a contextual retrieval query for follow-up questions
    retrieval_query = question

    previous_questions = [
        message[1]
        for message in chat_history
        if message[0] == "human"
    ]

    if previous_questions:
        retrieval_query = (
            f"Previous question: {previous_questions[-1]}\n"
            f"Follow-up question: {question}"
        )

    # Step 2: Retrieve candidate chunks
    candidate_chunks = retrive_chunks(
        retrieval_query,
        vectorstore,
        k=20
    )

    # Step 3: Rerank and select top 5
    retrieved_chunks = rerank_chunks(
        retrieval_query,
        candidate_chunks,
        top_k=5
    )

    if not retrieved_chunks:
        return {
            "answer": "Information not found in the transcripts.",
            "sources": [],
            "evidence_status": "INSUFFICIENT"
        }

    # Step 4: Verify evidence sufficiency
    evidence_status = check_evidence_sufficiency(
        retrieval_query,
        retrieved_chunks
    )

    if evidence_status != "SUFFICIENT":
        return {
            "answer": "Information not found in the transcripts.",
            "sources": [],
            "evidence_status": evidence_status
        }

    # Step 5: Format transcript context
    formatted_context = format_transcript_context(
        retrieved_chunks
    )

    # Step 6: Generate answer using conversation history
    chain = CHAT_PROMPT | model | StrOutputParser()

    response = chain.invoke({
        "context": formatted_context,
        "question": question,
        "chat_history": chat_history
    })

    return {
        "answer": response,
        "sources": retrieved_chunks,
        "evidence_status": evidence_status
    }