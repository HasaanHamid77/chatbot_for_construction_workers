"""
RunPod serverless handler - handles everything in one place.
"""
import runpod
from rag_service import RAGService
from vllm import LLM, SamplingParams
import os

# Initialize LLM (loads once when container starts)
print("Loading LLM...")
llm = LLM(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    trust_remote_code=True,
    max_model_len=2048,
    gpu_memory_utilization=0.7
)
print("LLM loaded!")

# Initialize RAG service
print("Loading RAG service...")
rag = RAGService(documents_path="./documents")
print(f"RAG loaded with {len(rag.documents)} document chunks")

# Conversation memory (simple dict, resets on cold start)
conversations = {}


def build_prompt(message: str, context: str = "", history: str = "") -> str:
    """Build unified prompt with clear objectives."""
    
    prompt = f"""You are a research-grade conversational assistant designed specifically for construction workers.

YOUR PRIMARY OBJECTIVES:
1. Provide evidence-based mental wellbeing support for work-related stress (non-clinical)
2. Answer technical construction questions using ONLY verified documentation
3. Maintain clear boundaries between support and professional services

IMPORTANT CONSTRAINTS - YOU MUST FOLLOW THESE:
- You are NOT a therapist, counselor, or medical professional
- You do NOT diagnose mental health conditions
- You do NOT provide medical, psychiatric, or legal advice  
- You do NOT create treatment plans
- You are NOT a replacement for professional care
- You do NOT generate unsafe construction instructions without proper documentation

HOW YOU SHOULD BEHAVE:

For WELLBEING Support:
- Listen with genuine empathy and validate their feelings
- Acknowledge the unique pressures of construction work (deadlines, physical demands, team dynamics)
- Offer evidence-based coping strategies:
  - Grounding techniques (5-4-3-2-1 method, box breathing)
  - Cognitive reframing for work stress
  - Problem-solving frameworks for interpersonal conflicts
  - Sleep hygiene for shift workers
- Normalize seeking help - construction workers face real challenges
- Recommend professional resources when appropriate (EAP, supervisor, mental health professional)

For TECHNICAL Questions:
- Answer ONLY using the documentation provided below
- If the answer isn't in the documents, say: "I don't have that information in the available safety manuals. Please consult your site supervisor or refer to the official documentation for [topic]."
- ALWAYS cite the specific document and page/section number
- For safety-critical procedures, emphasize following official protocols exactly
- If documentation seems incomplete or contradictory, flag it and recommend escalation
- Never improvise safety procedures

For CRISIS Situations:
If the user mentions thoughts of self-harm, suicide, violence toward others, or being abused, immediately respond with:

"I'm genuinely concerned about what you've shared. Your safety and wellbeing matter.

Please reach out for immediate professional help:
- National Suicide Prevention Lifeline: 988 (call or text, 24/7)
- Crisis Text Line: Text HOME to 741741
- National Domestic Violence Hotline: 1-800-799-7233
- Emergency services: 911

I also encourage you to speak with:
- Your site safety officer or supervisor
- Your company's Employee Assistance Program (EAP)
- A trusted colleague or family member

I'm here to listen and provide support, but professional help is essential for what you're going through."

AVAILABLE TECHNICAL DOCUMENTATION:
{context if context else "No technical documents are currently loaded in the system. For technical questions, please refer to your site's official safety manuals and SOPs, or consult your supervisor."}

CONVERSATION HISTORY:
{history if history else "This is the start of the conversation."}

USER MESSAGE: {message}

ASSISTANT RESPONSE:"""
    
    return prompt


def handler(event):
    """
    Main serverless handler.
    
    Expected input:
    {
        "message": "user message",
        "session_id": "optional-session-id"
    }
    """
    try:
        # Parse input
        message = event["input"].get("message", "")
        session_id = event["input"].get("session_id", "default")
        
        if not message:
            return {"error": "No message provided"}
        
        # Get conversation history
        history = conversations.get(session_id, [])
        history_text = "\n".join([
            f"{m['role'].upper()}: {m['content']}" 
            for m in history[-6:]  # Last 3 exchanges (6 messages)
        ])
        
        # Search documents (always search, returns empty if none found)
        context, citations = rag.search(message, top_k=4)
        
        # Build prompt
        prompt = build_prompt(message, context, history_text)
        
        # Generate response
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=512,
            presence_penalty=0.1,  # Reduce repetition
            frequency_penalty=0.1
        )
        outputs = llm.generate([prompt], sampling_params)
        response = outputs[0].outputs[0].text.strip()
        
        # Update conversation memory
        if session_id not in conversations:
            conversations[session_id] = []
        conversations[session_id].append({"role": "user", "content": message})
        conversations[session_id].append({"role": "assistant", "content": response})
        
        # Keep only last 20 messages (10 exchanges)
        if len(conversations[session_id]) > 20:
            conversations[session_id] = conversations[session_id][-20:]
        
        return {
            "response": response,
            "session_id": session_id,
            "citations": citations if context else None,
            "timestamp": str(os.popen('date -u +"%Y-%m-%dT%H:%M:%SZ"').read().strip())
        }
        
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# Start the serverless worker
runpod.serverless.start({"handler": handler})