# Summarizer AI API

A robust FastAPI-based summarization service specializing in local inference with GGUF models. Optimized for T5 architecture and CPU execution.

## 🚀 Features

- **T5 Architecture Support**: Correct manual encoder-decoder inference implementation for `llama.cpp`.
- **State Isolation**: Automatic context management ensures fresh state for every request, fixing common inference errors in T5.
- **Async API**: Non-blocking inference execution using FastAPI and `starlette.concurrency`.
- **Security**: Environment-based API Secret Key authentication (`X-API-Key` header).
- **Validation**: Strict Pydantic validation for all requests and token limits.
- **Detailed Metrics**: Accurate tracking of input/output tokens and processing time.
- **Thread-Safe**: Safely handles concurrent requests.

## 🛠 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Ozodbekk1/summarizer_ai.git
   cd summarizer_ai
   ```

2. **Set up a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## ⚙️ Configuration

Create a `.env` file based on `.env.example`:
```bash
API_SECRET_KEY=your_secure_key_here
MODEL_PATH=summarizer_ai_model.gguf
API_PORT=8000
API_HOST=0.0.0.0
```

## 🖥 Usage

### Starting the API
```bash
python3 main.py
```

### API Endpoints

- **GET `/health`**: Check service and model status.
- **POST `/summarize`**: Main summarization endpoint.

#### Request Body:
```json
{
  "text": "Your input text here",
  "instructions": "Summarize this in exactly 3 bullet points",
  "min_tokens": 10,
  "max_tokens": 128,
  "temperature": 0.0
}
```

#### Response Body:
```json
{
  "summary": "The generated summary...",
  "input_tokens": 45,
  "output_tokens": 30,
  "total_tokens": 75,
  "processing_time_ms": 150
}
```

## 🧪 Testing

Run the comprehensive test suite:
```bash
python3 tests/test_api.py
```

## ⚠️ Model Limitations & Alternatives

The current model `summarizer_ai_model.gguf` is a **T5-small (61M parameters)** architecture. 

**Limitations:**
- **Instruction Following**: Due to its small size and traditional T5 training (focused on the `summarize: ` prefix), it has very limited natural language instruction-following capabilities. It primarily performs extractive/abstractive summarization in a fixed style.
- **Multilingual Support**: While it can process various inputs, its output quality for languages like Uzbek or cross-lingual tasks (Uzbek to English) may be limited compared to larger models.

**Recommended Alternatives (GGUF):**
If your use case requires strict instruction following (e.g., "exactly 3 bullet points") or better multilingual support, consider replacing the model with:
- **Flan-T5-Base/Large**: Better instruction following while keeping the T5 architecture.
- **Phi-3-mini-4k-instruct (GGUF)**: Excellent instruction following for its size (3.8B).
- **Qwen2-1.5B-Instruct (GGUF)**: Very capable small model with great multilingual support.

## 📬 Postman Examples

1. **Health Check**:
   - Method: `GET`
   - URL: `{{base_url}}/health`

2. **Summarize Text**:
   - Method: `POST`
   - URL: `{{base_url}}/summarize`
   - Headers: `X-API-Key: {{secret_key}}`
   - Body (JSON):
     ```json
     {
       "text": "The Artificial Intelligence (AI) field has seen significant advancements in recent years...",
       "instructions": "One sentence summary",
       "max_tokens": 64
     }
     ```
