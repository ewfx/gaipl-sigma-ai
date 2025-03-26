import pandas as pd

# Optional: Only import transformers when needed
try:
    from transformers import pipeline
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
except ImportError:
    summarizer = None

def load_logs(logs_path):
    return pd.read_csv(logs_path)

def get_logs_for_trace_id(trace_id, logs_df):
    if not trace_id:
        return []
    logs = logs_df[logs_df["trace_id"] == trace_id]
    return logs["log"].tolist()

def summarize_logs(log_lines):
    if not summarizer or not log_lines:
        return "Summary not available (transformers not installed or no logs provided)."
    
    combined = "\n".join(log_lines)
    result = summarizer(combined, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
    
    # Deduplicate repeated sentences
    seen = set()
    summary_sentences = []
    for sent in result.split(". "):
        sent_clean = sent.strip()
        if sent_clean and sent_clean not in seen:
            seen.add(sent_clean)
            summary_sentences.append(sent_clean)
    
    return [f"{s.rstrip('.') + '.'}".strip("•○–- ").strip() for s in summary_sentences]