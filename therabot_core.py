from http import client
import json, os
from datetime import datetime
from dotenv import load_dotenv
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
load_dotenv()

client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MEMORY_FILE = "long_term_memory.json"

SYSTEM_PROMPT = """
You are TheraBot, an empathetic, therapist-like assistant.

Rules:
- Validate emotions first
- Ask gentle open-ended questions
- Never diagnose or prescribe
- Encourage professional help during severe distress
- Never claim to replace therapy
"""

CRISIS_WORDS = [
    "suicide", "kill myself", "end my life",
    "self harm", "cut myself", "no reason to live","die"
]

def crisis_detected(text):
    return any(w in text.lower() for w in CRISIS_WORDS)

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return []
    try:
        with open(MEMORY_FILE, "r") as f:
            content = f.read().strip()
            return json.loads(content) if content else []
    except json.JSONDecodeError:
        return []

def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

def get_client():
    return OpenAI()

def embed(client, text):
    return np.array(
        client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        ).data[0].embedding
    )

def recall_memory(client, query, memory, k=3):
    if not memory:
        return []
    q_emb = embed(client, query)
    scored = [
        (
            cosine_similarity([q_emb], [np.array(m["embedding"])])[0][0],
            m["text"]
        )
        for m in memory
    ]
    scored.sort(reverse=True)
    return [text for _, text in scored[:k]]

def get_response(conversation, user_input):
    client = get_client()
    memory = load_memory()

    if crisis_detected(user_input):
        return (
            "I'm really sorry you're feeling this much pain.\n\n"
            "You deserve real-world support.\n"
            "**India crisis helpline:** AASRA 24/7 – +91-9820466726\n\n"
            "If you're in immediate danger, please contact emergency services."
        )

    recalled = recall_memory(client, user_input, memory)
    memory_context = "\n".join(recalled)

    messages = conversation + [{"role": "user", "content": user_input}]
    if recalled:
        messages.append({
            "role": "system",
            "content": f"Relevant past context:\n{memory_context}"
        })

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.4,
        messages=messages
    )

    reply = response.choices[0].message.content

    if len(user_input.split()) > 8:
        memory.append({
            "timestamp": datetime.now().isoformat(),
            "text": user_input,
            "embedding": embed(client, user_input).tolist()
        })
        save_memory(memory)

    return reply


    return reply

