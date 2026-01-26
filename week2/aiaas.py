import os
import time
import json
import base64
import requests
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

PARAGRAPH = ("Artificial intelligence (AI) and cloud computing have rapidly evolved into mutually reinforcing pillars of modern digital ecosystems, transforming industries, accelerating research, and reshaping education, business, and software development. As organizations move beyond experimentation, AI is shifting from an add-on technology to a foundational layer embedded across applications, services, and operations. This transition is visible in enterprise environments, higher education, software engineering curricula, and global technology strategy. Cloud platforms provide the scalable compute, storage, networking, and distributed architectures needed for AI models to be trained, deployed, and continuously improved. In turn, AI enhances how cloud systems operate—optimizing resource allocation, automating security monitoring, enabling intent-based development, accelerating DevOps workflows, and powering intelligent analytics that help institutions and businesses make data-driven decisions. At the center of this technological convergence is a growing demand for systems that are scalable, cost-efficient, secure, and capable of supporting increasingly complex AI workloads.")

PROMPT = f"Summarize the following paragraph into 1 sentence, with the only punctuation used being a period: {PARAGRAPH}"

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
    model = os.getenv("HF_MODEL", "sshleifer/distilbart-cnn-12-6")  

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

    print("\nInput Paragraph:\n", PARAGRAPH, "\n")

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