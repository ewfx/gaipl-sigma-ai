# 📁 File: app/chatbot_router.py
from app.model_runner import generate_root_cause_analysis, generate_genai_response, describe_network, suggest_missing_connections
from app.log_checker import get_logs_for_trace_id, summarize_logs
from app.change_checker import get_related_changes
from app.vector_search import retrieve_similar_incidents
from app.network_viz import generate_dot
from app.intelscope import query_entry
import pandas as pd
import re

def extract_app_name(text, all_apps):
    return next((word for word in text.split() if word.isupper() and word in all_apps), None)

def check_unknown_targets(text, all_apps):
    return [word for word in text.split() if word.isupper() and word not in all_apps]

def extract_uuid(text):
    match = re.search(r"[a-f0-9]{8}-[a-f0-9]{4}-[1-5][a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}", text, re.I)
    return match.group(0) if match else None

def process_chatbot_query(query, df, model, index, logs_df, change_df, cmdb_df, network_data ,chat_history=None, kb_doc_id=None):
    if not query.strip():
        return "🤖 Please enter a question so I can help!"

    query_lower = query.lower()
    all_apps = [entry["app"] for entry in network_data]

    if any(keyword in query_lower for keyword in ["diagram", "architecture", "design"]):
        unknown_targets = check_unknown_targets(query, all_apps)
        if unknown_targets:
            return f"⚠️ Unknown system(s) mentioned: {', '.join(unknown_targets)}. Please check the system name(s)."

        app_name = extract_app_name(query, all_apps)
        if app_name:
            return generate_dot(app_name, network_data)
        else:
            return "Please mention a valid system name to generate the architecture diagram."
    if "root cause" in query_lower or "analysis" in query_lower or "rca" in query_lower:
        similar = retrieve_similar_incidents(query, model, index, df)
        return generate_root_cause_analysis(query, similar)
    
    if "incident" in query_lower and "inc" in query_lower:
        # Try to extract incident ID
        inc_id = next((word for word in query.split() if word.startswith("INC")), None)
        if inc_id and inc_id in df["incident_id"].values:
            row = df[df["incident_id"] == inc_id].iloc[0]
            changes = get_related_changes(row["app"], row["incident_date"], change_df)
            return f"Incident {inc_id}:\nDescription: {row['description']}\nCause: {row['cause']}\nResolution: {row['resolution']}\nRelated Changes: {len(changes)}"
        else:
            return "Incident ID not found."
        
    
    if "trace" in query_lower or "log" in query_lower or extract_uuid(query):
        trace_id = extract_uuid(query) or next((word for word in query.split() if word.startswith("trace") or word.startswith("log")), None)
        logs = get_logs_for_trace_id(trace_id, logs_df)
        if logs:
            summary = summarize_logs(logs)
            return "\n".join(summary)
        else:
            return "No logs found for that trace ID."

    # elif "network" in query_lower or "connect" in query_lower:
    #     return generate_genai_response(query_lower, network_data)

    # Step 2: Attempt context-based prompt chaining if the query is generic
    if any(word in query_lower for word in ["explain", "detail" , "describe"]):
        if chat_history:
            for past in reversed(chat_history):
                if past["role"] == "user":
                    recent_context = past["content"]
                    combined_prompt = f"User: {recent_context}\nUser: {query}"

                    # Route based on context + follow-up
                    if any(k in recent_context.lower() for k in ["diagram", "architecture", "design"]):
                        unknown_targets = check_unknown_targets(query, all_apps)
                        if unknown_targets:
                            return f"⚠️ Unknown system(s) mentioned: {', '.join(unknown_targets)}. Please check the system name(s)."

                        app_name = extract_app_name(recent_context, all_apps)
                        if app_name:
                            return generate_dot(app_name, network_data)
                        else:
                            return "Please mention a valid system name to generate the architecture diagram."
                    if any(k in recent_context.lower() for k in ["root cause","analysis","rca"]):
                        similar = retrieve_similar_incidents(query, model, index, df)
                        return generate_root_cause_analysis(query, similar)
                    

                    if any(k in recent_context.lower() for k in ["incident", "inc"]):
                        inc_id = next((word for word in recent_context.split() if word.startswith("INC")), None)
                        if inc_id and inc_id in df["incident_id"].values:
                            row = df[df["incident_id"] == inc_id].iloc[0]
                            changes = get_related_changes(row["app"], row["incident_date"], change_df)
                            return f"Incident {inc_id}:\nDescription: {row['description']}\nCause: {row['cause']}\nResolution: {row['resolution']}\nRelated Changes: {len(changes)}"

                    if any(k in recent_context.lower() for k in ["trace", "log"]) or extract_uuid(recent_context):
                        trace_id = extract_uuid(recent_context) or next((word for word in recent_context.split() if word.startswith("trace") or word.startswith("log")), None)
                        logs = get_logs_for_trace_id(trace_id, logs_df)
                        if logs:
                            return "\n".join(summarize_logs(logs))

                    #Default to GenAI network response.
                    unknown_targets = check_unknown_targets(combined_prompt, all_apps)
                    if unknown_targets:
                        return f"⚠️ Unknown system(s) mentioned: {', '.join(unknown_targets)}. Please check the system name(s)."

                    return generate_genai_response(combined_prompt, network_data)
    
    

    # Check for KB doc follow-up
    if kb_doc_id:
        kb_answer = query_entry(kb_doc_id, query)
        if kb_answer and "Content not found" not in kb_answer:
            return f"📖 Based on the uploaded document:\n\n{kb_answer}"


    

    else:
        return "🤖 I'm still learning! That question seems outside my scope right now."