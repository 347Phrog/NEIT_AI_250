import os
import time
import json
import base64
import requests
from google import genai
from dotenv import load_dotenv
from google.genai import types

load_dotenv()

user_input = input("Please give the AI text, within errors for it to fix: ")


PROMPT = f"Correct ANY and ALL grammatical errors in the following text: {user_input}"


def timed(label, fn):
    start = time.perf_counter()
    try:
        result = fn()
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {"provider": label, "ok": True, "latency_ms": round(elapsed_ms, 1), "result": result}
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {"provider": label, "ok": False, "latency_ms": round(elapsed_ms, 1), "error": str(e)}

def summarize_huggingface():
    token = os.getenv("HF_TOKEN")
    model = os.getenv("HF_MODEL", "pszemraj/led-large-book-summary")  

    url = f"https://router.huggingface.co/hf-inference/models/{model}"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": PROMPT}  

    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()

    if isinstance(data, list) and data and "summary_text" in data[0]:
        return data[0]["summary_text"]
    return data

def summarize_google():
   
    client = genai.Client()

    model = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash-001")
    response = client.models.generate_content(
        model=model,
        contents=PROMPT
    )
    return response.text

def main():
    tests = []

    if os.getenv("HF_TOKEN"):
        tests.append(("Hugging Face Summarization", summarize_huggingface))
    else:
        print("Skipping Hugging Face")

    if os.getenv("GOOGLE_API_KEY"):
        tests.append(("Google Gemini Summarization", summarize_google))
    else:
        print("Skipping Google")


    results = [timed(label, fn) for label, fn in tests]

    for r in results:
        print(f"== {r['provider']} ==")
        print(f"OK: {r['ok']} Latency: {r['latency_ms']} ms")
        if r["ok"]:
            print("Summary/Output:")
            print(r["result"] if isinstance(r["result"], str) else json.dumps(r["result"], indent=2))
        else:
            print("Error:", r["error"])
        print()

if __name__ == "__main__":
    main()
