from prompts import CROSS_ANALYSIS_PROMPT
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from rag_chain import answer_question
from transcript_parser import parse_transcript
from chunker import chunk_transcript
from vector_store import create_vector_store
import os
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(
        temperature=0.0,
        model_name="openai/gpt-oss-120b",
        groq_api_key=os.environ.get("GROQ_API_KEY")
        )

def analyze_cross_experts(question,vectorstore):
    """
    Gathers cited responses from all three experts for a given interview question,
    then uses CROSS_ANALYSIS_PROMPT to synthesize common themes and contrasts/disagreements.
    """
    experts = [
    {"name": "Dr. Jean Martin", "market": "France"},
    {"name": "Anna Keller", "market": "Germany"},
    {"name": "Dr. Emily Carter", "market": "United Kingdom"}
    ]

    expert_answers = []

    for expert in experts:
        current_expert = expert["name"]
        answer = answer_question(question, vectorstore, expert_name=current_expert)

        expert_answer = { "Expert"  : current_expert,"Answer" : answer }
        expert_answers.append(expert_answer)
    formatted_responses = "" 

    for item in expert_answers:
        expert_name = item["Expert"]
        text_answer = item["Answer"]["answer"] 
        formatted_responses += f"Expert: {expert_name}\nAnswer: {text_answer}\n\n"

    chain = CROSS_ANALYSIS_PROMPT | model | StrOutputParser()

    response = chain.invoke({
        "questions" : question,
        "expert_responses" :formatted_responses
    })

    return {
        "cross_analysis" : response,
        "individual_answers" : expert_answers
    }

if __name__ == "__main__":

    print("--- 1. Ingesting & Indexing Transcripts ---")

    files = [
    "data/transcripts/Transcript_1_France.txt",
    "data/transcripts/Transcript_2_Germany.txt",
    "data/transcripts/Transcript_3_UK.txt"
    ]
    all_chunks = []
    for f in files:
        h, s = parse_transcript(f)
        all_chunks.extend(chunk_transcript(h, s))

    vectorstore = create_vector_store(all_chunks)

    test_question = "What are the main barriers to adoption?"
    print(f"\n--- 2. Asking: '{test_question}' ---")

    # Call your answer_question function
    response = analyze_cross_experts(test_question, vectorstore)

    print("\n--- 3. LLM Response ---")
    print(response["cross_analysis"])

    print("\n--- 4. Sources Cited ---")
    
    for doc in response["individual_answers"]:
        print(doc)
        print("-----------------------------")