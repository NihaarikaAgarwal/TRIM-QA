#!/usr/bin/env python3
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

'''
Mistral 7b selected for this demo because it works well with QA and summarization of large data. Also support group query attetnion. 
'''
def create_context(chunks, max_length=400):
    """
    Joins chunk texts with newlines but stops if we exceed max_length tokens 
    It will summarize or truncate if thrshold is met
    """
    joined_texts = []
    current_length = 0
    for chunk in chunks:
        chunk_text = chunk["text"]
        chunk_words = len(chunk_text.split())
        if current_length + chunk_words > max_length:
            break
        joined_texts.append(chunk_text)
        current_length += chunk_words
    return "\n".join(joined_texts)

def format_prompt(chunks, question, max_context_tokens=400):
    """
    Builds the final prompt by combining the pruned/reranked chunks with the question.
    """
    context = create_context(chunks, max_length=max_context_tokens)
    return (
        f"Answer the question based strictly on the context below. Do not generate any extra text.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )


def main():
    #output from reranking 
    json_file = "/home/alozan12/TRIM-QA-old/test_LLM.json"
    with open(json_file, "r") as f:
        chunks = json.load(f)
    
    # question, use question test file later, each has exected answer for evaluation
    #begin loop here if necessary, looping between all questions 
    question = "What is each person's age?"
    
    # Build the prompt
    prompt = format_prompt(chunks, question, max_context_tokens=200)
    print("PROMPT:\n", prompt, "\n")
    
    # Load Mistral-7B-Instruct
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    token = "MistralToken"  # Replace your token
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token, torch_dtype=torch.float16, device_map="auto")
    print("Model loaded.\n")
    
    # tokenize the prompt and generate the answer
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.1
    )
    
    # Decode and post-process the output
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\n", answer)

if __name__ == "__main__":
    main()
