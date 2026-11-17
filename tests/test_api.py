import sys
import os
import threading
import time
import json
from fastapi.testclient import TestClient

# Add parent directory to path to import api
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import app

client = TestClient(app)

SECRET_KEY = "my_super_secret_key"
HEADERS = {"X-Secret-Key": SECRET_KEY}

def log_result(name, summary, metrics):
    with open("test_summary_output.txt", "a") as f:
        f.write(f"\n=== {name} ===\n")
        f.write(f"Summary: {summary}\n")
        f.write(f"Metrics: {json.dumps(metrics)}\n")

def test_health():
    print("Testing /health...")
    response = client.get("/health")
    assert response.status_code == 200
    print("Health OK:", response.json())

def test_basic_summarization():
    print("\nTesting basic summarization...")
    payload = {
        "text": "The Artificial Intelligence (AI) field has seen significant advancements in recent years. Large Language Models (LLMs) have become particularly prominent, demonstrating remarkable capabilities in understanding and generating human-like text.",
        "min_tokens": 10,
        "max_tokens": 50
    }
    response = client.post("/summarize", json=payload, headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    log_result("Basic", data["summary"], {k: v for k, v in data.items() if k != "summary"})

def test_instruction_one_sentence():
    print("\nTesting one-sentence instruction...")
    payload = {
        "text": "The Artificial Intelligence (AI) field has seen significant advancements in recent years. Large Language Models (LLMs) have become particularly prominent, demonstrating remarkable capabilities in understanding and generating human-like text.",
        "instructions": "Summarize this in exactly one sentence.",
        "max_tokens": 50
    }
    response = client.post("/summarize", json=payload, headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    log_result("One Sentence", data["summary"], {k: v for k, v in data.items() if k != "summary"})

def test_instruction_bullet_points():
    print("\nTesting bullet points instruction...")
    payload = {
        "text": "The Artificial Intelligence (AI) field has seen significant advancements in recent years. Large Language Models (LLMs) have become particularly prominent, demonstrating remarkable capabilities in understanding and generating human-like text. GGUF is a file format used for storing models.",
        "instructions": "Summarize this in exactly 3 bullet points.",
        "max_tokens": 100
    }
    response = client.post("/summarize", json=payload, headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    log_result("Bullet Points", data["summary"], {k: v for k, v in data.items() if k != "summary"})

def test_uzbek_input():
    print("\nTesting Uzbek input...")
    payload = {
        "text": "Sun'iy intellekt (AI) sohasi so'nggi yillarda katta yutuqlarga erishdi. Katta til modellari (LLM) ayniqsa mashhur bo'lib, inson kabi matnni tushunish va yaratishda ajoyib qobiliyatlarni namoyish etdi.",
        "instructions": "Summarize this Uzbek text in English.",
        "max_tokens": 100
    }
    response = client.post("/summarize", json=payload, headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    log_result("Uzbek", data["summary"], {k: v for k, v in data.items() if k != "summary"})

def test_invalid_auth():
    print("\nTesting invalid authentication...")
    payload = {"text": "Test"}
    response = client.post("/summarize", json=payload, headers={"X-Secret-Key": "wrong"})
    assert response.status_code == 403
    print("Invalid auth correctly rejected.")

def test_concurrent_requests():
    print("\nTesting concurrent requests...")
    results = []
    
    def make_request(i):
        payload = {
            "text": f"Request {i}: " + "The quick brown fox jumps over the lazy dog. " * 5,
            "max_tokens": 20
        }
        res = client.post("/summarize", json=payload, headers=HEADERS)
        results.append(res.status_code)
        print(f"Request {i} finished with {res.status_code}")

    threads = [threading.Thread(target=make_request, args=(i,)) for i in range(3)]
    for t in threads: t.start()
    for t in threads: t.join()
    
    assert all(code == 200 for code in results)
    print("Concurrent requests finished successfully.")

def test_validation_errors():
    print("\nTesting validation errors...")
    # Empty text
    res = client.post("/summarize", json={"text": ""}, headers=HEADERS)
    assert res.status_code == 422
    
    # Invalid token limits
    res = client.post("/summarize", json={"text": "test", "min_tokens": 100, "max_tokens": 50}, headers=HEADERS)
    assert res.status_code == 422
    print("Validation errors correctly handled.")

if __name__ == "__main__":
    with TestClient(app) as client:
        test_health()
        test_basic_summarization()
        test_instruction_one_sentence()
        test_instruction_bullet_points()
        test_uzbek_input()
        test_invalid_auth()
        test_concurrent_requests()
        test_validation_errors()
    print("\nAll tests completed.")
