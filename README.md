### Summarizer AI

A lightweight FastAPI-based summarization service powered by a T5-small model in GGUF format. This project provides both a web API and a standalone script for text summarization with support for custom instructions and token constraints.

#### Features

- **FastAPI Web Service**: High-performance API for summarization.
- **T5 GGUF Model**: Efficient inference using `llama-cpp-python` for encoder-decoder architecture.
- **Security**: Mandatory `X-Secret-Key` header for API protection.
- **Customizable**: Control `min_tokens`, `max_tokens`, and provide custom `instructions` for each summary.
- **Validation**: Built-in token limit checks to ensure model stability.

#### Installation

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

#### Configuration

Create a `.env` file in the root directory (based on `.env.example`):
```bash
API_SECRET_KEY=your_custom_secret_key
```

#### Usage

##### Running the API

Start the FastAPI server:
```bash
python3 api.py
```
The server will be available at `http://0.0.0.0:8000`.

##### API Endpoints

- **POST `/summarize`**
  - **Headers**:
    - `x-secret-key`: Your secret key (default: `my_super_secret_key`)
    - `Content-Type`: `application/json`
  - **Body**:
    ```json
    {
      "text": "Your long text to summarize...",
      "min_tokens": 20,
      "max_tokens": 128,
      "instructions": "Focus on technical details."
    }
    ```

- **GET `/health`**
  - Returns the health status of the service.

##### Standalone Script

You can also run summarization directly via the CLI:
```bash
python3 summarize.py
```

#### Development

- `api.py`: The FastAPI application and summarization logic.
- `summarize.py`: A standalone script for quick testing.
- `summarizer_ai_model.gguf`: The T5-small model weights.

#### License

MIT
