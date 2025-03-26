# # 📁 File: app/intelscope.py
# import pandas as pd
# import os
# import uuid
# import datetime
# from app.model_runner import generate_genai_response

# KB_FILE = "data/knowledgebase.csv"


# def load_knowledgebase():
#     if os.path.exists(KB_FILE):
#         return pd.read_csv(KB_FILE)
#     else:
#         return pd.DataFrame(columns=["id", "source", "content", "added_on"])


# def save_to_knowledgebase(source, content):
#     kb = load_knowledgebase()
#     entry = {
#         "id": str(uuid.uuid4()),
#         "source": source,
#         "content": content,
#         "added_on": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     }
#     kb = pd.concat([kb, pd.DataFrame([entry])], ignore_index=True)
#     kb.to_csv(KB_FILE, index=False)
#     return entry


# def list_sources():
#     kb = load_knowledgebase()
#     return kb[["id", "source", "added_on"]]


# def get_content_by_id(doc_id):
#     kb = load_knowledgebase()
#     row = kb[kb["id"] == doc_id]
#     return row.iloc[0]["content"] if not row.empty else None


# def summarize_entry(doc_id):
#     content = get_content_by_id(doc_id)
#     if content:
#         return generate_genai_response(f"Summarize the following content:\n{content}", [])
#     return "Content not found."


# def query_entry(doc_id, question):
#     content = get_content_by_id(doc_id)
#     if content:
#         prompt = f"Answer the following question based on this content:\n{content}\n\nQuestion: {question}"
#         return generate_genai_response(prompt, [])
#     return "Content not found."


from transformers import pipeline
import pandas as pd
import os
import uuid
import datetime

KB_FILE = "data/knowledgebase.csv"

# Initialize the BART summarizer pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

def summarize_text_bart(content):
    try:
        summary_chunks = []
        max_input = 1024
        step = 800
        for i in range(0, len(content), step):
            chunk = content[i:i + max_input]
            summary = summarizer(chunk, max_length=200, min_length=50, do_sample=False)[0]["summary_text"]
            summary_chunks.append(summary)
        return "\n".join(summary_chunks)
    except Exception as e:
        return f"⚠️ Summarization failed: {str(e)}"

def load_knowledgebase():
    if os.path.exists(KB_FILE):
        return pd.read_csv(KB_FILE)
    else:
        return pd.DataFrame(columns=["id", "source", "content", "added_on"])

def save_to_knowledgebase(source, content):
    kb = load_knowledgebase()
    entry = {
        "id": str(uuid.uuid4()),
        "source": source,
        "content": content,
        "added_on": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    kb = pd.concat([kb, pd.DataFrame([entry])], ignore_index=True)
    kb.to_csv(KB_FILE, index=False)
    return entry

def list_sources():
    kb = load_knowledgebase()
    return kb[["id", "source", "added_on"]]

def get_content_by_id(doc_id):
    kb = load_knowledgebase()
    row = kb[kb["id"] == doc_id]
    return row.iloc[0]["content"] if not row.empty else None

def summarize_entry(doc_id):
    content = get_content_by_id(doc_id)
    if content:
        return summarize_text_bart(content)
    return "Content not found."

def query_entry(doc_id, question):
    content = get_content_by_id(doc_id)
    if content:
        from app.model_runner import generate_genai_response
        prompt = f"Answer the following question based on this content:\n{content}\n\nQuestion: {question}"
        return generate_genai_response(prompt, [])
    return "Content not found."
