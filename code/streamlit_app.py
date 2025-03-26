import torch
import types
import streamlit as st
import json
import pandas as pd
from app.data_loader import load_incident_data
from app.vector_search import build_vector_index, retrieve_similar_incidents
from app.model_runner import generate_root_cause_analysis,generate_genai_response,describe_network,suggest_missing_connections,answer_rca_question
from app.change_checker import get_related_changes, load_changes, load_cmdb
from app.log_checker import get_logs_for_trace_id, load_logs, summarize_logs
from app.network_viz import generate_dot
from app.intelscope import save_to_knowledgebase, summarize_entry, query_entry
import docx2txt
from PyPDF2 import PdfReader

def extract_text_from_file(uploaded_file):
    filetype = uploaded_file.name.split(".")[-1].lower()

    if filetype == "txt":
        return uploaded_file.read().decode("utf-8", errors="ignore")

    elif filetype == "pdf":
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    elif filetype == "docx":
        return docx2txt.process(uploaded_file)

    return ""
# Patch torch error
torch.classes.__path__ = types.SimpleNamespace(_path=[])

st.set_page_config(page_title="Integrated Platform Environment (IPE) Analyzer", layout="wide")
st.title("🔎 IPE - Integrated Platform Environment")
# --- Load Data ---
data_path = "data/incident_data.csv"
df = load_incident_data(data_path)
index, embeddings, model = build_vector_index(df)
cmdb_df = load_cmdb("data/CMDB_Mapping.csv")
change_df = load_changes("data/change.csv")
logs_df = load_logs("data/Logs_Lookup.csv")
network_df = pd.read_csv("data/network_metadata.csv")
import ast
network_df["api_flows"] = network_df["api_flows"].apply(ast.literal_eval)

network_data = network_df.to_dict(orient="records")

# --- Card Selection ---
# st.markdown("##")

# Session state keys to isolate selections
if "active_card" not in st.session_state:
    st.session_state.active_card = None

if "show_chatbot" not in st.session_state:
    st.session_state.show_chatbot = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "reset_chat_input" not in st.session_state:
    st.session_state.reset_chat_input = False


if "chat_input_value" not in st.session_state:
    st.session_state.chat_input_value = ""

if "last_uploaded_filename" not in st.session_state:
    st.session_state["last_uploaded_filename"] = ""


import streamlit as st

# Custom CSS for cards and buttons
st.markdown("""
<style>
.card-wrapper {
    display: flex;
    justify-content: space-evenly;
    margin-top: 30px;
    gap: 20px;
    padding-bottom: 60px;
}

.card {
    background: #f2f2f2;
    border-radius: 18px;
    padding: 20px 20px 40px;
    width: 90%;
    max-width: 380px;
    min-height: 150px;
    border: 1px solid #333;
    box-shadow: inset 0 0 0.5px rgba(255, 255, 255, 0.1);
    text-align: center;
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: space-between;
    overflow: visible;
}
.card::after {
    content: "";
    position: absolute;
    bottom: 0;
    width: 60%;
    height: 1px;
    background: rgba(255,255,255,0.05);
}
.stButton > button {
    position: absolute;
    bottom: -25px;
    left: 45%;
    transform: translateX(-50%);
    width: 70%;
    padding: 0.65rem 1.1rem;
    font-size: 1.1rem;
    border-radius: 12px;
    border: 1px solid #555;
    background: linear-gradient(135deg, #3a3a3a, #1f1f1f);
    color: #f5f5f5;      
    font-weight: 500;
    cursor: pointer;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4);
    z-index: 2;
    transition: all 0.25s ease;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #4a4a4a, #2a2a2a);
    border-color: #888;
    box-shadow: 0 6px 14px rgba(0, 0, 0, 0.5);
}

.card-desc {
    font-size: 15px;
    color: #333;
    font-style: italic;
    flex-grow: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
}
.small-button .stButton > button {
    padding: 0.25rem 0.6rem;
    font-size: 0.8rem;
    width: auto;
    border-radius: 6px;
    line-height: 1.2;
    min-height: 20px;
    margin-bottom: 2rem; /* 🔥 This adds vertical spacing */
}
.chat-entry {
    background-color: #222;
    color: white;
    padding: 0.5rem;
    margin-bottom: 0.5rem;
    border-radius: 10px;
}
.vertical-divider {
    border-left: 2px solid #aaa;
    height: 10cm;
    margin: 0 10px;
}

</style>
""", unsafe_allow_html=True)


# Function to reset card state
def reset_card_state():
    st.session_state.active_card = None
    for key in ["rca_result", "rca_text_source", "rca_similar_data", "incident_selected", "logs_data", "chat_history", "last_asked", "chat_mia_input"]:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.show_chatbot = False

# Initialize session state if not already set
if "active_card" not in st.session_state:
    st.session_state.active_card = None

# Card and button layout
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="card">
        <div class="card-desc">Delve into issues with deeper insights, discover co-related change requests enhanced by GenAI-powered RCA.</div>
        <div class="stButtonContainer" id="smart-button"></div>
    </div>
    """, unsafe_allow_html=True)
    btn = st.button("🧠 Smart Issue Explorer", key="smart")
    if btn:
        reset_card_state()
        st.session_state.active_card = "smart"

with col2:
    st.markdown("""
    <div class="card">
        <div class="card-desc">Effortlessly extract insights from documents or links with instant GenAI responses.</div>
        <div class="stButtonContainer" id="intelscope-button"></div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("📘 GenieDocs", key="intelscope"):
        reset_card_state()
        st.session_state.active_card = "intelscope"

with col3:
    st.markdown("""
    <div class="card">
        <div class="card-desc">Gain deeper understanding of system behavior through GenAI-driven log summarization.</div>
        <div class="stButtonContainer" id="smart-button"></div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🧬 TraceIQ", key="tracer"):
        reset_card_state()
        st.session_state.active_card = "tracer"

with col4:
    st.markdown("""
    <div class="card">
        <div class="card-desc">Visualize and explore your app's full network — now with GenAI insights, RCA queries, and smart suggestions.</div>
        <div class="stButtonContainer" id="smart-button"></div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🌐 NetViz Explorer", key="netviz"):
        reset_card_state()
        st.session_state.active_card = "netviz"



# st.markdown("""
# <div class="card">
#     <div class="card-desc">Talk to Mia!, Your GenAI assistant to explore incidents, logs, and network queries in a natural conversation.</div>
#     <div class="stButtonContainer" id="chat-launch-button"></div>
# </div>
# """, unsafe_allow_html=True)
st.markdown('##')
if st.button("💬 Chat with Dora!", key="genai_chat"):
    reset_card_state()
    st.session_state.show_chatbot = True
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_asked" not in st.session_state:
        st.session_state.last_asked = ""
st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# --- Smart Issue Explorer ---
if st.session_state.active_card == "smart":
    st.subheader("🧠 Smart Issue Explorer")
    query = st.text_input("Enter issue or symptom description:")

    if query:
        incident_row = df[df["incident_id"] == query]
        query_text = (
            incident_row.iloc[0]["combined_text"] if not incident_row.empty else query
        )

        with st.spinner("Retrieving similar incidents..."):
            similar = retrieve_similar_incidents(query_text, model, index, df)
            st.dataframe(similar[["incident_id", "description", "resolution", "cause"]])
        
        st.markdown('<div class="small-button">', unsafe_allow_html=True)
        if st.button("Analyze"):
            st.session_state.rca_text_source = query_text
            st.session_state.rca_similar_data = similar
            with st.spinner("Generating Root Cause Analysis..."):
                st.session_state.rca_result = generate_root_cause_analysis(
                    st.session_state.rca_text_source,
                    st.session_state.rca_similar_data
                )
        st.markdown('</div>', unsafe_allow_html=True)

        # Show RCA if previously generated
        if "rca_result" in st.session_state and st.session_state.active_card == "smart":
            st.markdown("### 🧠 Probable Root Cause and Resolution")
            st.markdown(st.session_state.rca_result)
        st.markdown("---")  # adds a horizontal line
        st.markdown("### 🔍 Explore Further")

        if st.checkbox("🔍 Do you want to dig deeper into either of these incidents?", value=False):
            for i, row in similar.iterrows():
                col1, col2 = st.columns([6, 1])
                with col1:
                    st.write(f"**{row['incident_id']}** | {row['description']} | {row['cause']}")
                with col2:
                    st.markdown('<div class="small-button">', unsafe_allow_html=True)
                    if st.button("Review", key=f"review_issue_{i}"):
                        st.session_state["incident_selected"] = row.to_dict()
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

# --- IntelScope ---
elif st.session_state.active_card == "intelscope":
    st.subheader("📘 GenieDocs")
    st.markdown("Upload a document or share a link. Ask questions to get instant GenAI-powered insights.")

    uploaded_file = st.file_uploader("Upload a file (PDF, TXT, DOCX)", type=["pdf", "txt", "docx"])
    url_input = st.text_input("Or paste a link to summarize")

    if uploaded_file or url_input:
        st.success("✅ Content successfully added to knowledgebase.")
        content = extract_text_from_file(uploaded_file) if uploaded_file else ""  # (or handle url_input here)
        entry = save_to_knowledgebase(uploaded_file.name if uploaded_file else url_input, content)
        doc_id = entry["id"]
        summary = summarize_entry(doc_id)
        st.markdown(summary)

        st.markdown("You can now ask a question below:")
        user_q = st.text_input("Ask something about the uploaded content")
        if user_q:
            st.markdown("Answering your question...")
            answer = query_entry(doc_id, user_q)
            st.markdown(f"**IntelScope:** {answer}")

# --- TraceIQ ---
elif st.session_state.active_card == "tracer":
    # st.subheader("🧬 TraceIQ")
    # trace_input = st.text_input("Enter Trace ID to investigate:")
    # if trace_input:
    #     logs = get_logs_for_trace_id(trace_input, logs_df)
    #     if logs:
    #         st.markdown(f"**🧠 Log Summary for Trace ID: `{trace_input}`**")
    #         summary = summarize_logs(logs)
    #         for line in summary:
    #             clean_line = line.strip("•○–- ").strip()
    #             st.markdown(f"- {clean_line}")
    #     else:
    #         st.info("No logs found for this trace ID.")
    st.subheader("🧬 TraceIQ")

    trace_input = st.text_input("Enter Trace ID to investigate:")
    selected_app = st.selectbox("Optional: Select the App related to this Trace ID (for better context)", [""] + sorted(cmdb_df["app"].unique()))

    if trace_input:
        if selected_app:
            data_sources = cmdb_df[cmdb_df["app"] == selected_app]["data_source"].unique()
            source_display = ", ".join(data_sources)
            st.markdown(f"ℹ️ Gathering logs from **{source_display}**...")
            logs = get_logs_for_trace_id(trace_input, logs_df)
        else:
            st.markdown("🔍 Checking logs across all data sources...")
            logs = get_logs_for_trace_id(trace_input, logs_df)

        if logs:
            st.markdown(f"**🧠 Log Summary for Trace ID: `{trace_input}`**")
            summary = summarize_logs(logs)
            for line in summary:
                clean_line = line.strip("•○–- ").strip()
                st.markdown(f"- {clean_line}")
        else:
            st.info("No logs found for this trace ID.")


elif st.session_state.active_card == "netviz":
    st.subheader("🌐 NetViz Explorer")
    app_names = [entry["app"] for entry in network_data]
    selected_app = st.selectbox("Select an App", app_names)
    api_filter = st.text_input("Filter by API name (optional)")

    if selected_app:
        dot_graph = generate_dot(selected_app,network_data, api_filter)
        st.graphviz_chart(dot_graph)
    with st.expander("💬 Ask GenAI about this network"):
        st.markdown("You can ask for descriptions, RCA insights, or suggest improvements.")
        nl_query = st.text_input("Enter a question (e.g., 'Why would SGA fail if EFG is down?')", key="genai_network_q")

        if nl_query.strip():
            with st.spinner("Thinking..."):
                result = answer_rca_question(nl_query.strip(), selected_app, network_data)
                st.markdown(result)

    with st.expander("📄 Auto Description"):
        # if st.button("Generate Description"):
        with st.spinner("Describing..."):
            st.session_state.auto_description = describe_network(selected_app, network_data)
        if "auto_description" in st.session_state:
            st.markdown(st.session_state.auto_description)

    with st.expander("🧩 Suggest Missing Connections"):
        with st.spinner("Thinking..."):
            st.session_state.suggestions = suggest_missing_connections(selected_app, network_data)
        if "suggestions" in st.session_state:
            st.markdown(st.session_state.suggestions)

# --- Shared RCA Panels ---
if "incident_selected" in st.session_state:
    incident = st.session_state["incident_selected"]

    with st.expander("📋 Incident Review (IR)", expanded=True):
        st.write(f"**Incident ID:** {incident['incident_id']}")
        st.write(f"**Description:** {incident['description']}")
        st.write(f"**App:** {incident['app']} ({incident['app_name']})")
        st.write(f"**Date:** {incident['incident_date']}")
        st.write(f"**Cause:** {incident['cause']}")
        st.write(f"**Resolution:** {incident['resolution']}")

    with st.expander("🔁 Change Review (CR)", expanded=False):
        st.write("Looking for related Change Requests...")
        app = incident['app']
        incident_date = incident['incident_date']
        related_crs = get_related_changes(app, incident_date, change_df)

        def correlation_weight(cr_date):
            try:
                return abs((pd.to_datetime(incident_date) - pd.to_datetime(cr_date)).days)
            except:
                return 999

        if related_crs:
            related_crs = sorted(related_crs, key=lambda x: correlation_weight(x["date"]))
            top_related = related_crs[:5]
            selected_cr = incident.get('cr_number', '')

            for cr in top_related:
                cr_date = pd.to_datetime(cr["date"])
                inc_date = pd.to_datetime(incident_date)
                days_diff = abs((inc_date - cr_date).days)

                if selected_cr:
                    status = '🔴 Strong correlation' if cr['cr_number'] == selected_cr else '🟠 Partial correlation'
                else:
                    status = '🟢 All clear' if days_diff > 3 else ('🟠 Partial correlation' if days_diff > 1 else '🔴 Strong correlation')

                ci_list = cmdb_df[cmdb_df["app"] == cr["app"]]["ci_id"].tolist()
                ci_text = ", ".join(ci_list)
                app_name = cr["app"]

                st.markdown(
                    f"""🔹 **CR:** `{cr['cr_number']}`  
📅 **Date:** `{cr['date']}`  
🧩 **App:** `{app_name}`  
🧮 **CI(s):** `{ci_text}`  
📊 **Correlation:** {status}"""
                )
                st.markdown("---")
        else:
            st.info("No related Change Requests found.")

    with st.expander("📄 Logs Check (LC)", expanded=False):
        trace_id = incident.get("trace_id", "")
        if trace_id:
            logs = get_logs_for_trace_id(trace_id, logs_df)
            if logs:
                st.markdown(f"**🧠 Log Summary for Trace ID: `{trace_id}`**")
                summary = summarize_logs(logs)
                for line in summary:
                    clean_line = line.strip("•○–- ").strip()
                    st.markdown(f"- {clean_line}")
            else:
                st.info("No logs found for this trace ID.")
        else:
            st.info("No trace ID associated with this incident.")

from app.chatbot import process_chatbot_query
if st.session_state.show_chatbot:
    with st.container():
        st.markdown("#### Hi! This is Dora 🕵️‍♀️, *Your Explorer*🗺️, How may I help you?")
        from app.intelscope import save_to_knowledgebase, summarize_entry,query_entry

        uploaded_file = st.file_uploader("📄 Upload a file to IntelScope (PDF, TXT, DOCX)", type=["txt", "pdf", "docx"])

        if uploaded_file:
            import docx2txt
            from PyPDF2 import PdfReader

            file_type = uploaded_file.type
            if file_type == "text/plain":
                content = uploaded_file.read().decode("utf-8")
            elif file_type == "application/pdf":
                pdf = PdfReader(uploaded_file)
                content = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
            elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                content = docx2txt.process(uploaded_file)
            else:
                content = ""

            if content:
                if st.session_state.get("last_uploaded_filename") != uploaded_file.name:
                    entry = save_to_knowledgebase(uploaded_file.name, content)
                    st.session_state["last_uploaded_doc_id"] = entry["id"]
                    st.session_state["last_uploaded_filename"] = uploaded_file.name
                    summary = summarize_entry(entry["id"])
                    st.session_state.chat_history.append({
                        "role": "bot",
                        "content": f"📄 **File '{uploaded_file.name}' uploaded.**\n\n**Summary:**\n{summary}"
                    })
                    st.rerun()


            if st.session_state.get("last_uploaded_doc_id"):
                st.markdown("🧠 You can ask questions about the last uploaded document.")

        for entry in st.session_state.chat_history:
            if entry["role"] == "user":
                st.markdown(f"<div class='chat-entry'><strong>You:</strong> {entry['content']}</div>", unsafe_allow_html=True)
            else:
                if "digraph {" in entry["content"]:
                    dot_start = entry["content"].find("digraph {")
                    intro_text = entry["content"][:dot_start].strip()
                    dot_code = entry["content"][dot_start:]
                    if intro_text:
                        st.markdown(f"<div class='chat-entry'><strong>🕵️‍♀️:</strong> {intro_text}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='chat-entry'><strong>🕵️‍♀️:</strong></div>", unsafe_allow_html=True)
                    st.graphviz_chart(dot_code)
                else:
                    st.markdown(f"<div class='chat-entry'><strong>🕵️‍♀️:</strong> {entry['content']}</div>", unsafe_allow_html=True)
        chat_input = st.text_input("Ask your question:", key="chat_mia_input")

        if chat_input and st.session_state.get("last_asked", "") != chat_input:
            st.session_state.chat_history.append({"role": "user", "content": chat_input})
            st.session_state.last_asked = chat_input

            with st.spinner("Dora is thinking..."):
                chat_result = process_chatbot_query(
                    chat_input,
                    df=df,
                    model=model,
                    index=index,
                    logs_df=logs_df,
                    change_df=change_df,
                    cmdb_df=cmdb_df,
                    network_data=network_data,
                    chat_history=st.session_state.chat_history,
                    kb_doc_id=st.session_state.get("last_uploaded_doc_id")
                )
                st.session_state.chat_history.append({"role": "bot", "content": chat_result})

                # Clear the widget by triggering a rerun with the same key but no value
                del st.session_state["chat_mia_input"]
                st.rerun()