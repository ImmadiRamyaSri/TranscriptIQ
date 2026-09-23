from langchain_groq import ChatGroq
from typing import List
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from transcript_parser import parse_transcript
from chunker import chunk_transcript
from vector_store import create_vector_store,retrive_chunks
import os
from dotenv import load_dotenv
load_dotenv()
from prompts import ANSWER_QUESTION_PROMPT

model = ChatGroq(
    temperature=0.0,
    model_name="openai/gpt-oss-120b",
    groq_api_key=os.environ.get("GROQ_API_KEY")
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

def answer_question(question, vectorstore, expert_name=None):
    retrived_chunks = retrive_chunks(question, vectorstore, expert_name=expert_name)
    formated_chunks = format_transcript_context(retrived_chunks)

    chain = ANSWER_QUESTION_PROMPT | model | StrOutputParser()

    response = chain.invoke({
    "context": formated_chunks, 
    "question": question
    })
    return {
        "answer":response,
        "sources":retrived_chunks
    }


if __name__ == "__main__":

    print("--- 1. Ingesting & Indexing Transcripts ---")
    header_data, segments = parse_transcript("data/transcripts/Transcript_1_France.txt")
    chunks = chunk_transcript(header_data, segments)
    vectorstore = create_vector_store(chunks)

    test_question = "What is the typical hospital decision-making timeline for purchasing a new robotic system?"
    print(f"\n--- 2. Asking: '{test_question}' ---")

    # Call your answer_question function
    response = answer_question(test_question, vectorstore)

    print("\n--- 3. LLM Response ---")
    print(response["answer"])

    print("\n--- 4. Sources Cited ---")
    for doc in response["sources"]:
        print(f"- {doc.metadata.get('expert_name')} ({doc.metadata.get('timestamp')}): {doc.page_content[:80]}...")