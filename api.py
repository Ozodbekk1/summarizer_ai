from fastapi import FastAPI, HTTPException, Header, Depends, Body
from pydantic import BaseModel
import llama_cpp
import ctypes
import numpy as np
import os
from typing import Optional

app = FastAPI(title="Summarizer AI API")

# Configuration
MODEL_PATH = "summarizer_ai_model.gguf"
SECRET_KEY = os.getenv("API_SECRET_KEY", "my_super_secret_key")
MAX_INPUT_TOKENS = 400  # Leave some room for the model's 512 context

class Summarizer:
    def __init__(self, model_path):
        print(f"Loading model: {model_path}...")
        self.llm = llama_cpp.Llama(model_path=model_path, n_ctx=512, verbose=False, embedding=True)
        self.eos_id = self.llm.token_eos()
        self.pad_id = 0  # T5 pad token is typically 0

    def get_token_count(self, text: str) -> int:
        tokens = self.llm.tokenize(text.encode('utf-8'), add_bos=False)
        return len(tokens)

    def summarize(self, text: str, min_tokens: int = 20, max_tokens: int = 128, instructions: Optional[str] = None):
        # Clear KV cache to avoid state carry-over
        mem = llama_cpp.llama_get_memory(self.llm.ctx)
        if mem:
            llama_cpp.llama_memory_clear(mem, True)

        # Build prompt with optional instructions
        prompt = "summarize: "
        if instructions:
            prompt += instructions.strip() + " "
        prompt += text
            
        tokens = self.llm.tokenize(prompt.encode('utf-8'), add_bos=False)
        if tokens[-1] != self.eos_id:
            tokens.append(self.eos_id)
        n_tokens = len(tokens)
        
        if n_tokens > 512:
            raise ValueError(f"Input too long: {n_tokens} tokens (max 512)")

        # Manual encoder step for T5 architecture
        batch = llama_cpp.llama_batch_init(n_tokens, 0, 1)
        try:
            batch.n_tokens = n_tokens
            for i in range(n_tokens):
                batch.token[i] = tokens[i]
                batch.pos[i] = i
                batch.n_seq_id[i] = 1
                batch.seq_id[i][0] = 0
                batch.logits[i] = 1 # Set all to 1 for encoder
            
            ret = llama_cpp.llama_encode(self.llm.ctx, batch)
            if ret != 0:
                return f"Error: llama_encode failed with code {ret}"
        finally:
            llama_cpp.llama_batch_free(batch)
        
        # Decoder start
        decoder_token_id = self.pad_id
        output_tokens = []
        
        for i in range(max_tokens):
            token_arr = (llama_cpp.llama_token * 1)(decoder_token_id)
            batch_d = llama_cpp.llama_batch_get_one(token_arr, 1, i, 0)
            
            ret = llama_cpp.llama_decode(self.llm.ctx, batch_d)
            if ret != 0:
                break
            
            logits_ptr = llama_cpp.llama_get_logits(self.llm.ctx)
            n_vocab = self.llm.n_vocab()
            logits = np.ctypeslib.as_array(logits_ptr, shape=(n_vocab,))
            
            # Prevent EOS if under min_tokens
            if len(output_tokens) < min_tokens:
                logits[self.eos_id] = -1e10
            
            # Greedy sampling
            next_token = int(np.argmax(logits))
            
            if next_token == self.eos_id:
                break
                
            output_tokens.append(next_token)
            decoder_token_id = next_token
            
        return self.llm.detokenize(output_tokens).decode('utf-8', errors='ignore').strip()

# Initialize summarizer
summarizer = Summarizer(MODEL_PATH)

class SummaryRequest(BaseModel):
    text: str
    min_tokens: Optional[int] = 10
    max_tokens: Optional[int] = 128
    instructions: Optional[str] = None

class SummaryResponse(BaseModel):
    summary: str
    tokens_used: int

async def verify_secret_key(x_secret_key: str = Header(...)):
    if x_secret_key != SECRET_KEY:
        raise HTTPException(status_code=403, detail="Invalid Secret Key")
    return x_secret_key

@app.post("/summarize", response_model=SummaryResponse)
async def api_summarize(
    request: SummaryRequest, 
    secret_key: str = Depends(verify_secret_key)
):
    token_count = summarizer.get_token_count(request.text)
    
    if token_count > MAX_INPUT_TOKENS:
        raise HTTPException(
            status_code=400, 
            detail=f"Input text too long: {token_count} tokens. Limit is {MAX_INPUT_TOKENS}."
        )
    
    try:
        summary = summarizer.summarize(
            request.text, 
            min_tokens=request.min_tokens, 
            max_tokens=request.max_tokens,
            instructions=request.instructions
        )
        return SummaryResponse(summary=summary, tokens_used=token_count)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
