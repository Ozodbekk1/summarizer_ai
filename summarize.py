import llama_cpp
import numpy as np
import time
import logging
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Summarizer:
    def __init__(self, model_path: str, n_ctx: int = 512):
        self.model_path = model_path
        self.n_ctx = n_ctx
        # We load a temporary instance just to get some metadata
        llm = llama_cpp.Llama(model_path=model_path, n_ctx=1, verbose=False)
        self.eos_token_id = llm.token_eos()
        self.bos_token_id = llm.token_bos()
        self.vocab_size = llm.n_vocab()
        del llm

    def _sample(self, logits, temperature=0.7, top_p=0.9):
        if temperature <= 0:
            return int(np.argmax(logits))
        
        logits = logits / temperature
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        
        indices_to_remove = cumulative_probs > top_p
        indices_to_remove[1:] = indices_to_remove[:-1]
        indices_to_remove[0] = False
        
        probs[sorted_indices[indices_to_remove]] = 0
        probs = probs / (np.sum(probs) + 1e-10)
        
        return int(np.random.choice(len(probs), p=probs))

    def summarize(
        self, 
        text: str, 
        instructions: Optional[str] = None,
        min_tokens: int = 10, 
        max_tokens: int = 128,
        temperature: float = 0.0
    ):
        start_time = time.time()
        
        # Load fresh instance for each request to ensure state isolation (T5 requirement)
        # and thread safety. Given the small model size (60MB), this is fast (~60ms).
        llm = llama_cpp.Llama(
            model_path=self.model_path, 
            n_ctx=self.n_ctx, 
            verbose=False, 
            embedding=True
        )
        
        try:
            prompt = "summarize: "
            if instructions:
                prompt += instructions.strip() + ": "
            prompt += text.strip()

            tokens = llm.tokenize(prompt.encode('utf-8'), add_bos=False)
            if not tokens or tokens[-1] != self.eos_token_id:
                tokens.append(self.eos_token_id)
            
            input_token_count = len(tokens)
            
            # Encoder
            n_tokens = len(tokens)
            batch = llama_cpp.llama_batch_init(n_tokens, 0, 1)
            try:
                batch.n_tokens = n_tokens
                for i in range(n_tokens):
                    batch.token[i] = tokens[i]
                    batch.pos[i] = i
                    batch.n_seq_id[i] = 1
                    batch.seq_id[i][0] = 0
                    batch.logits[i] = True
                
                ret = llama_cpp.llama_encode(llm.ctx, batch)
                if ret != 0:
                    raise RuntimeError(f"Encoder failed: {ret}")
            finally:
                llama_cpp.llama_batch_free(batch)

            # Decoder
            output_tokens = []
            curr_token_id = 0 # T5 decoder start token
            for i in range(max_tokens):
                t_arr = (llama_cpp.llama_token * 1)(curr_token_id)
                b_d = llama_cpp.llama_batch_get_one(t_arr, 1, i, 0)
                
                ret = llama_cpp.llama_decode(llm.ctx, b_d)
                if ret != 0: break
                
                logits_ptr = llama_cpp.llama_get_logits(llm.ctx)
                logits = np.ctypeslib.as_array(logits_ptr, shape=(self.vocab_size,)).copy()
                
                if len(output_tokens) < min_tokens:
                    logits[self.eos_token_id] = -1e10
                
                next_token = self._sample(logits, temperature=temperature)
                if next_token == self.eos_token_id: break
                
                output_tokens.append(next_token)
                curr_token_id = next_token
                
            summary_text = llm.detokenize(output_tokens).decode('utf-8', errors='ignore').strip()
            
            return {
                "summary": summary_text,
                "input_tokens": input_token_count,
                "output_tokens": len(output_tokens),
                "total_tokens": input_token_count + len(output_tokens),
                "processing_time_ms": int((time.time() - start_time) * 1000)
            }
        finally:
            # Llama instance is cleaned up when it goes out of scope or explicitly deleted
            del llm

    def tokenize(self, text: str) -> List[int]:
        # Temporary instance for tokenization
        llm = llama_cpp.Llama(model_path=self.model_path, n_ctx=1, verbose=False)
        tokens = llm.tokenize(text.encode('utf-8'), add_bos=False)
        return tokens
