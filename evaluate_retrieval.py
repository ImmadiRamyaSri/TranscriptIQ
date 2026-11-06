import json
import csv
from transcript_parser import parse_transcript
from chunker import chunk_transcript
from rag_chain import check_evidence_sufficiency, rerank_chunks
from vector_store import create_vector_store, retrive_chunks


if __name__ == "__main__":

    # Step 1: Load and parse transcripts
    print("\n--- 1. Loading Transcripts ---")

    files = [
        "data/transcripts/Transcript_1_France.txt",
        "data/transcripts/Transcript_2_Germany.txt",
        "data/transcripts/Transcript_3_UK.txt"
    ]

    all_chunks = []

    for file_path in files:
        header, segments = parse_transcript(file_path)
        chunks = chunk_transcript(header, segments)
        all_chunks.extend(chunks)

    print(f"Total transcript chunks: {len(all_chunks)}")


    # Step 2: Create FAISS vector store
    print("\n--- 2. Creating FAISS Vector Store ---")

    vectorstore = create_vector_store(all_chunks)

    print(f"Total indexed documents: {vectorstore.index.ntotal}")


    # Step 3: Load evaluation questions
    print("\n--- 3. Loading Evaluation Questions ---")

    json_path = "data/evaluation/evaluation_questions.json"

    with open(json_path, "r", encoding="utf-8") as file:
        evaluation_data = json.load(file)

    print(f"Total evaluation questions: {len(evaluation_data)}")


    # Step 4: Retrieve top 20 chunks for each question
    print("\n--- 4. Evidence Sufficiency Evaluation ---")

    evaluation_results = []

    for item in evaluation_data:

        question_id = item["id"]
        question = item["question"]
        expected = item["expected_relevance"]

        # Step 1: Retrieve candidate chunks
        candidate_chunks = retrive_chunks(
            question,
            vectorstore,
            k=20
        )

        # Step 2: Rerank top 5 chunks
        retrieved_chunks = rerank_chunks(
            question,
            candidate_chunks,
            top_k=5
        )

        # Step 3: Check evidence sufficiency
        if not retrieved_chunks:
            evidence_status = "INSUFFICIENT"
        else:
            evidence_status = check_evidence_sufficiency(
                question,
                retrieved_chunks
            )

        # Step 4: Compare expected and actual results
        expected_status = (
            "SUFFICIENT"
            if expected == "relevant" or expected is True
            else "INSUFFICIENT"
        )

        evaluation_results.append({
            "question_id": question_id,
            "question": question,
            "expected_status": expected_status,
            "actual_status": evidence_status,
            "correct": expected_status == evidence_status
        })

    for result in evaluation_results:
        print(
            f"{result['question_id']} | "
            f"Expected: {result['expected_status']} | "
            f"Actual: {result['actual_status']} | "
            f"Correct: {result['correct']}"
        )

    correct = sum(r["correct"] for r in evaluation_results)
    total = len(evaluation_results)

    print(f"\nAccuracy: {correct}/{total} ({correct / total * 100:.2f}%)")


    output_path = "data/evaluation/evidence_sufficiency_results.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=evaluation_results[0].keys()
        )

        writer.writeheader()
        writer.writerows(evaluation_results)

    print(f"\nEvaluation results saved to: {output_path}")