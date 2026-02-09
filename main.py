from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
from handler import startup, generate, rag

app = FastAPI(title="Construction Assistant API")

# Conversation memory (simple in-memory storage)
conversations = {}


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"


class Citation(BaseModel):
    document: str
    page: Optional[int] = None
    text: str
    relevance_score: float


class ChatResponse(BaseModel):
    response: str
    session_id: str
    citations: Optional[List[Citation]] = None


@app.on_event("startup")
def load():
    """Load model and RAG on startup."""
    startup()


@app.get("/")
def health():
    """Health check endpoint."""
    return {
        "status": "running",
        "model": "Qwen2.5-1.5B-Instruct",
        "documents_loaded": len(rag.documents) if rag else 0
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Main chat endpoint with RAG integration.
    
    Automatically:
    1. Retrieves relevant documents
    2. Maintains conversation history
    3. Generates response with proper context
    """
    
    message = request.message
    session_id = request.session_id
    
    if not message:
        return ChatResponse(
            response="Please provide a message.",
            session_id=session_id,
            citations=None
        )
    
    # Get conversation history for this session
    history = conversations.get(session_id, [])
    history_text = "\n".join([
        f"{msg['role'].upper()}: {msg['content']}" 
        for msg in history[-6:]  # Last 3 exchanges
    ])
    
    # Search for relevant documents using RAG
    context, citations = rag.search(message, top_k=4)
    
    # Generate response with context and history
    response_text = generate(
        user_message=message,
        context=context,
        history=history_text
    )
    
    # Update conversation memory
    if session_id not in conversations:
        conversations[session_id] = []
    
    conversations[session_id].append({
        "role": "user",
        "content": message
    })
    conversations[session_id].append({
        "role": "assistant",
        "content": response_text
    })
    
    # Keep only last 20 messages (10 exchanges)
    if len(conversations[session_id]) > 20:
        conversations[session_id] = conversations[session_id][-20:]
    
    # Format citations for response
    formatted_citations = None
    if citations:
        formatted_citations = [
            Citation(
                document=cite['document'],
                page=cite.get('page'),
                text=cite['text'],
                relevance_score=cite['relevance_score']
            )
            for cite in citations
        ]
    
    return ChatResponse(
        response=response_text,
        session_id=session_id,
        citations=formatted_citations
    )


@app.post("/clear_session/{session_id}")
def clear_session(session_id: str):
    """Clear conversation history for a specific session."""
    if session_id in conversations:
        del conversations[session_id]
        return {"status": "cleared", "session_id": session_id}
    return {"status": "not_found", "session_id": session_id}


@app.get("/sessions")
def list_sessions():
    """List all active sessions."""
    return {
        "active_sessions": list(conversations.keys()),
        "total": len(conversations)
    }


@app.get("/stats")
def get_stats():
    """Get system statistics."""
    return {
        "documents_loaded": len(rag.documents) if rag else 0,
        "active_sessions": len(conversations),
        "total_messages": sum(len(conv) for conv in conversations.values())
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)