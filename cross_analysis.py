from prompts import CROSS_ANALYSIS_PROMPT
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from rag_chain import answer_question
from vector_store import get_available_experts
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
    Dynamically discovers experts, retrieves individual answers,
    and synthesizes common themes and disagreements.
    """
    # Discover experts from vector store metadata
    experts = get_available_experts(vectorstore)
 
    expert_answers = []

    for expert in experts:
        current_expert = expert["name"]
        answer = answer_question(question, vectorstore, expert_name=current_expert)

        expert_answer = { 
            "Expert"  : current_expert,
            "Market": expert["market"],
            "Role": expert["role"],
            "Answer": answer
            }
        expert_answers.append(expert_answer)
        formatted_responses = ""

        for item in expert_answers:
            formatted_responses += (
                f"Expert: {item['Expert']}\n"
                f"Market: {item['Market']}\n"
                f"Role: {item['Role']}\n"
                "Original Transcript Evidence:\n"
            )

            for i, doc in enumerate(item["Answer"]["sources"], start=1):
                formatted_responses += (
                    f"\n[Source {i}]\n"
                    f"Expert: {doc.metadata.get('expert_name', item['Expert'])}\n"
                    f"Market: {doc.metadata.get('market', item['Market'])}\n"
                    f"Timestamp: {doc.metadata.get('timestamp', 'Unknown')}\n"
                    f"Transcript: {doc.page_content}\n"
                )

            formatted_responses += "\n" + "-" * 50 + "\n\n"
    
    chain = CROSS_ANALYSIS_PROMPT | model | StrOutputParser()

    response = chain.invoke({
        "questions" : question,
        "expert_responses" :formatted_responses
    })

    return {
        "cross_analysis" : response,
        "individual_answers" : expert_answers
    }