def chunk_transcript( header_data , segments ):
    """
    Pairs interviewer questions with expert responses into unified semantic chunks.
    Attaches expert metadata and the expert's timestamp for precise retrieval citations.
    """
    chunks = []
    
    for i in range(0,len(segments),2):
        if i + 1 >= len(segments):
            break 
        chunk = {
                "text":None,
                "metadata": { 
                    "expert_name" : None, 
                    "role" : None, 
                    "market" : None, 
                    "timestamp" : None
                    }
                }
        
        chunk["text"] = segments[i]["text"] +" "+ segments[i + 1]["text"]
        chunk["metadata"]["expert_name"]  = header_data["name"]
        chunk["metadata"]["role"]         = header_data["role"]
        chunk["metadata"]["market"]       = header_data["market"]
        chunk["metadata"]["timestamp"]    = segments[i+1]["timestamp"]
        chunks.append(chunk)
    return chunks