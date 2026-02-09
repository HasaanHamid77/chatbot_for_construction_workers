import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rag_service import RAGService

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = None
model = None
rag = None

def startup():
    global tokenizer, model, rag

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("Model loaded")

    print("Loading RAG...")
    rag = RAGService("./documents")
    print("RAG ready")


def generate(user_message: str):
    prompt = f"""<|system|>
{SYSTEM_PROMPT}
<|user|>
{user_message}
<|assistant|>
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("<|assistant|>")[-1].strip()
