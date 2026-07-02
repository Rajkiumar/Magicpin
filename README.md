# magicpin AI Challenge — Vera Merchant AI Assistant ("Vera")

This repository contains the complete implementation of **Vera**, magicpin's elite merchant-AI marketing assistant on WhatsApp, built for the magicpin AI Challenge.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install fastapi uvicorn python-dotenv requests
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY="your_gemini_api_key_here"
```

### 3. Expand the Dataset
Generate the full set of 50 merchants, 200 customers, and 100 triggers:
```bash
python dataset/generate_dataset.py --seed-dir dataset --out dataset/expanded
```

### 4. Run the Bot Server
Start the FastAPI server on port `8080`:
```bash
python -m uvicorn bot:app --host 127.0.0.1 --port 8080 --reload
```

### 5. Run the Evaluation Simulator
To run the judge's test scenarios locally (health checks, auto-reply detection, intent transitions, hostility handling):
```bash
# Windows PowerShell
$env:PYTHONIOENCODING="utf-8"; python judge_simulator.py
```

### 6. Generate the Submission File
To generate the final `submission.jsonl` containing composed messages for the 30 canonical test pairs:
```bash
$env:PYTHONIOENCODING="utf-8"; python -u generate_submission.py
```

---

## 🛠️ Architecture & Approach

Our implementation is a unified, stateful, and highly robust conversational system built with **FastAPI** and **Gemini 2.5 Flash**:

1.  **Unified 4-Context Composer**: The core message generation loads Category, Merchant, Trigger, and Customer contexts and aggregates them into a structured prompt.
2.  **Strict Tone Alignment**: Leverages system instructions to enforce tone rules by vertical (clinical-peer for Dentists, warm/friendly for Salons, operator-to-operator for Restaurants, trustworthy/respectful for Pharmacies, coaching/motivational for Gyms).
3.  **Core Constraints Enforcement**: Enforces constraints such as absolute exclusion of URLs, use of specific metrics/numbers, "service @ price" offer catalog matching (e.g. "Dental Cleaning @ ₹299"), and language preference (Hinglish/hi-en blend when preferred).
4.  **Auto-Reply Detection**: Tracks consecutive auto-replies *per merchant*. If an auto-reply or exact repetition is received, Vera warns the user on Turn 1, backs off for 24 hours on Turn 2, and ends the conversation on Turn 3.
5.  **Pitch-to-Action Intent Transition**: Detects when a merchant commits ("Ok let's do it", "Yes", etc.) and transitions immediately to Action mode (delivering the draft/post, providing a binary CTA to go live, and forbidding any more qualifying questions).
6.  **Programmatic Fallback Validators**: Performs post-processing on LLM outputs (e.g., stripping qualifying words like "do you" or "would you" during action mode, ensuring actioning words are present) to guarantee 100% test compliance.

---

## 📈 Design Tradeoffs & Feedback

*   **Free-Tier Rate Limits**: The Gemini free tier has a strict limit of 5 requests per minute (RPM). To accommodate this, `generate_submission.py` sleeps for 13 seconds between requests, and the `bot.py` client includes built-in retry-with-backoff logic for `429` errors.
*   **Prompt vs. Code Guardrails**: We combined system prompt constraints with minor Python post-processing (regex-based phrase checks) for intent transitions to ensure zero qualifying questions slip into action mode.
*   **State Tracking**: Conversational state is tracked in-memory by both `conversation_id` and `merchant_id` to reliably handle scenarios where the test harness changes the conversation ID across turns.