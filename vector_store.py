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

def retrive_chunks(query,vectorstore,expert_name = None ,k=5):
    """
    Retrieves the most semantically relevant chunks for a query.
    Supports optional metadata filtering by expert name.
    """
    fetch_k = 20 if expert_name else 6

    similar_chunks= vectorstore.similarity_search(query, k=fetch_k)

    if expert_name is None:
        return similar_chunks[:k]

    matching_chunks = []
    for doc in similar_chunks:
        if doc.metadata['expert_name']  == expert_name:
            matching_chunks.append(doc)

    return matching_chunks[:k]