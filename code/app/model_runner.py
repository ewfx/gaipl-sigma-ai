from transformers import pipeline
from openai import OpenAI

# ✅ Use a better CPU-friendly model
hf_model = pipeline(
    "text2text-generation",
    model="MBZUAI/LaMini-Flan-T5-783M",
    device=-1  # CPU
)

client = OpenAI(OpenAI.api_key)

def huggingface_generate_response(prompt):
    print("🔄 Generating response using Hugging Face model...")
    truncated = prompt[:1100]  # LaMini is more compact, this is usually enough
    result = hf_model(
        truncated,
        max_length=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    output = result[0]['generated_text'].strip()
    return remove_repetitions(output)

def remove_repetitions(text):
    lines = text.splitlines()
    seen = set()
    clean_lines = []
    for line in lines:
        if line.strip() not in seen:
            clean_lines.append(line)
            seen.add(line.strip())
    return "\n".join(clean_lines)

def generate_root_cause_analysis(query, similar_df):
    summary = "\n".join([
        f"Incident {row['incident_id']}: {row['description']} | Resolution: {row['resolution']} | Cause: {row['cause']}"
        for _, row in similar_df.head(3).iterrows()
    ])

    if len(summary) > 800:
        summary = summary[:800] + "..."

    few_shot_example = """
Example:

Issue:
Intermittent connectivity to backend

Similar Incidents:
Incident INC1234: DNS failure | Resolution: Restarted DNS pod | Cause: DNS misconfiguration  
Incident INC1235: Timeout errors | Resolution: Increased timeout | Cause: Network congestion  
Incident INC1236: API failures | Resolution: Rebooted backend | Cause: stale routing table

---
**Probable Root Cause**:
Backend connectivity was failing due to a combination of stale DNS and unoptimized routing paths under network congestion.

**Resolution Plan**:
1. Restart backend and DNS-related pods to refresh stale entries.
2. Validate and update routing table configurations.
3. Monitor traffic load and implement timeout threshold adjustments.

**Preventive Suggestions**:
1. Set up alerts for increased packet drops or timeout errors.
2. Automate stale route detection via periodic probes.
---
"""

    prompt = f"""
You are a highly skilled SRE assistant. Based on the following issue and similar incidents, infer the root cause, recommend a 3-step resolution plan, and provide 2 preventive suggestions.
Even if individual incidents differ slightly, identify any **common failure patterns** that could explain the issue. Avoid rejecting incidents unless they are clearly unrelated.

Your output should **not copy any one incident**. Look for patterns and summarize.

{few_shot_example}

Now apply the same logic to this:

### Issue:
{query}

### Similar Incidents:
{summary}

Respond in the same format as the example:
"""


    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"⚠️ OpenAI call failed: {e}")
        print("⏪ Falling back to Hugging Face model...")
        return huggingface_generate_response(prompt)
    

# ---------------------------- GENAI NETWORK HELPERS -----------------------------

def generate_genai_response(prompt, network_data):
    dot_text = "\n".join([
        f"App: {entry['app']} connects to {[api['connects_to'] for api in entry['api_flows']]}"
        for entry in network_data
    ])
    context = f"Here is a network topology:\n{dot_text}\n"

    full_prompt = f"{context}\nQuestion: {prompt}\nAnswer with clarity and structure."

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.4
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI failed: {e}, falling back to Hugging Face")
        return huggingface_generate_response(full_prompt)

def describe_network(app_name, network_data):
    deps = next((entry for entry in network_data if entry["app"] == app_name), None)
    if not deps:
        return "No network data available."
    deps_list = [api['connects_to'] for api in deps["api_flows"]]
    prompt = f"Describe in one paragraph what the system '{app_name}' does, given it connects to: {', '.join(deps_list)}"
    return generate_genai_response(prompt, network_data)

def suggest_missing_connections(app_name, network_data):
    all_apps = [entry["app"] for entry in network_data if entry["app"] != app_name]
    prompt = f"What other systems might '{app_name}' need to connect to that are not currently defined?"
    return generate_genai_response(prompt, network_data)

def answer_rca_question(question, app_name, network_data):
    app_entry = next((entry for entry in network_data if entry["app"] == app_name), None)
    deps_list = [api['connects_to'] for api in app_entry["api_flows"]] if app_entry else []
    all_apps = [entry["app"] for entry in network_data]

    # Detect unknown targets in the user question
    unknown_targets = [word for word in question.split() if word.isupper() and word not in all_apps]
    if unknown_targets:
        return f"⚠️ Unknown system(s) mentioned: {', '.join(unknown_targets)}. Please check the system name(s)."

    context = (
        f"'{app_name}' is connected to: {', '.join(deps_list)}. "
        f"Known apps in the network are: {', '.join(all_apps)}. "
        "If the question involves unknown systems or invalid references, respond accordingly."
    )

    full_question = f"{context}\n\n{question}"
    return generate_genai_response(full_question, network_data)
