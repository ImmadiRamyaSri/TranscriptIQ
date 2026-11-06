from transcript_parser import parse_transcript
from chunker import chunk_transcript
from vector_store import create_vector_store
from rag_chain import answer_question


if __name__ == "__main__":

    # Step 1: Load transcripts
    print("\n--- Loading Transcripts ---")

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

    # Step 2: Create vector store
    print("\n--- Creating FAISS Vector Store ---")

    vectorstore = create_vector_store(all_chunks)

    print(f"Indexed documents: {vectorstore.index.ntotal}")

    # Step 3: Select test questions
    test_questions = [
        {
            "id": "Q1",
            "question": "What factors influence the adoption of robotic surgery?"
        },
        {
            "id": "Q22",
            "question": "How is robotic surgery adoption progressing in the United States?"
        },
        {
            "id": "Q24",
            "question": "What is the average purchase price of a robotic surgery system in India?"
        }
    ]

    # Step 4: Test complete RAG pipeline
    print("\n--- Testing Complete RAG Pipeline ---")

    for item in test_questions:

        print("\n" + "=" * 70)
        print(f"Question ID: {item['id']}")
        print(f"Question: {item['question']}")

        result = answer_question(
            item["question"],
            vectorstore
        )

        print("\nEvidence Status:")
        print(result["evidence_status"])

        print("\nGenerated Answer:")
        print(result["answer"])

        print("\nSources:")

        for doc in result["sources"]:
            print(
                f"- Expert: {doc.metadata.get('expert_name')}, "
                f"Market: {doc.metadata.get('market')}, "
                f"Timestamp: {doc.metadata.get('timestamp')}"
            )

    print("\n--- Testing Completed ---")
    