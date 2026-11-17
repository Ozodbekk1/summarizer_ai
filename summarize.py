import llama_cpp
import ctypes
import numpy as np

print("started")

def summarize_text(text, model_path="summarizer_ai_model.gguf", min_tokens=20, max_tokens=128, instructions=None):
    print(f"Loading model: {model_path}...")
    # Load model using the Llama class for convenience
    llm = llama_cpp.Llama(model_path=model_path, n_ctx=512, verbose=False, embedding=True)
    
    # Clear KV cache to avoid state carry-over (if instance was reused)
    mem = llama_cpp.llama_get_memory(llm.ctx)
    if mem:
        llama_cpp.llama_memory_clear(mem, True)
    
    # Build prompt with optional instructions
    prompt = "summarize: "
    if instructions:
        prompt += instructions.strip() + " "
    prompt += text
    tokens = llm.tokenize(prompt.encode('utf-8'), add_bos=False)
    if tokens[-1] != llm.token_eos():
        tokens.append(llm.token_eos())
    n_tokens = len(tokens)
    
    print(f"Encoding {n_tokens} tokens...")
    # Manual encoder step for T5 architecture
    batch = llama_cpp.llama_batch_init(n_tokens, 0, 1)
    try:
        batch.n_tokens = n_tokens
        for i in range(n_tokens):
            batch.token[i] = tokens[i]
            batch.pos[i] = i
            batch.n_seq_id[i] = 1
            batch.seq_id[i][0] = 0
            batch.logits[i] = True # For encoder, we need all outputs
        
        ret = llama_cpp.llama_encode(llm.ctx, batch)
        if ret != 0:
            return f"Error: llama_encode failed with code {ret}"
    finally:
        llama_cpp.llama_batch_free(batch)
    
    print("Decoding...")
    # Decoder start token for T5 is typically 0 (<pad>)
    decoder_token_id = 0
    output_tokens = []
    
    eos_id = llm.token_eos()
    for i in range(max_tokens):
        # Decode one token at a time
        token_arr = (llama_cpp.llama_token * 1)(decoder_token_id)
        batch_d = llama_cpp.llama_batch_get_one(token_arr, 1, i, 0)
        
        ret = llama_cpp.llama_decode(llm.ctx, batch_d)
        if ret != 0:
            print(f"llama_decode failed with code {ret}")
            break
        
        # Get logits and sample greedily
        logits_ptr = llama_cpp.llama_get_logits(llm.ctx)
        n_vocab = llm.n_vocab()
        logits = np.ctypeslib.as_array(logits_ptr, shape=(n_vocab,))
        
        # Prevent EOS if under min_tokens
        if len(output_tokens) < min_tokens:
            logits[eos_id] = -1e10
            
        # Greedy sampling
        next_token = int(np.argmax(logits))
        
        if next_token == eos_id:
            break
            
        output_tokens.append(next_token)
        decoder_token_id = next_token
        
    return llm.detokenize(output_tokens).decode('utf-8', errors='ignore').strip()

if __name__ == "__main__":
    sample_text = """
    The Artificial Intelligence (AI) field has seen significant advancements in recent years. 
    Large Language Models (LLMs) have become particularly prominent, demonstrating 
    remarkable capabilities in understanding and generating human-like text. 
    These models are trained on vast amounts of data and can perform various tasks 
    including translation, summarization, and creative writing. 
    GGUF is a file format used for storing models for inference with GGML and 
    llama.cpp-based executors. It is designed to be extensible and efficient.
    """
    
    summary = summarize_text(
        sample_text, 
        min_tokens=30, 
        instructions="Focus on the core concept and technical specifications."
    )
    print("\n--- Summary ---")
    print(summary)
    print("---------------")
