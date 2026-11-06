from langchain_core.documents import Document 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_vector_store(chunks):
    """
    Embeds transcript chunks using all-MiniLM-L6-v2 and indexes them into a FAISS vector store.
    """
    documents = []
    for chunk in chunks:
        doc = Document(
            page_content = chunk["text"],
            metadata = chunk["metadata"]
        )   
        documents.append(doc)

    model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents,model)

    return vectorstore


def retrive_chunks(query, vectorstore, expert_name=None, k=20):
    """
    Retrieves semantically relevant transcript chunks.
    Supports optional metadata filtering by expert name.
    """

    total_chunks = vectorstore.index.ntotal

    if total_chunks == 0:
        return []

    # If no expert is selected, retrieve the top k chunks.
    if expert_name is None:
        results = vectorstore.similarity_search(
            query,
            k=min(k,total_chunks)
        )
    else:
        # Apply metadata filtering during LangChain retrieval.
        results = vectorstore.similarity_search(
            query,
            k=min(k, total_chunks),
            filter={"expert_name": expert_name},
            fetch_k=total_chunks
            )

    return results


def get_available_experts(vectorstore):
    """
    Dynamically discovers experts and their metadata
    from the documents stored in FAISS.
    """

    experts = {}

    for doc_id in vectorstore.index_to_docstore_id.values():

        doc = vectorstore.docstore.search(doc_id)

        if not isinstance(doc, Document):
            continue

        metadata = doc.metadata
        expert_name = metadata.get("expert_name")

        if not expert_name:
            continue

        experts[expert_name] = {
            "name": expert_name,
            "market": metadata.get("market", "Unknown"),
            "role": metadata.get("role", "Unknown")
        }

    return sorted(
        experts.values(),
        key=lambda expert: expert["name"]
    )    