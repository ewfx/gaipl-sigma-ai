
# 🤖 GenAI-Powered Integrated Platform Environment

![Agentic AI](https://img.shields.io/badge/Powered_by-Agentic_AI-blueviolet)
![Context-Aware](https://img.shields.io/badge/Contextual-Recommendations-blue)
![Streamlit](https://img.shields.io/badge/Built_with-Streamlit-orange)

**Sigma-AI** is a **GenAI-enabled** Integrated Platform Environment (IPE) that empowers platform engineers by reducing context switching across tools. It brings observability, RCA generation, network analysis, and intelligent insights into a unified interface using vector search, CMDB correlations, and LLM-powered workflows.

---

## 📑 Table of Contents

1. [Introduction](#-introduction)
2. [Demo](#-demo)
3. [User-Interface Overview](#️-user-interface-overview)
4. [Multi-Model Strategy – Smart Fallback Mechanism](#-multi-model-strategy--smart-fallback-mechanism)
5. [Sigma-AI Solution Architecture Overview](#-sigma-ai-solution-architecture-overview)
6. [Incident Similarity Search with L2 (Euclidean) Distance](#-incident-similarity-search-with-l2-euclidean-distance)
7. [Intelligence Stack](#-intelligence-stack)
8. [Project Structure](#-project-structure)
9. [System Architecture](#system-architecture)
11. [Installation](#-installation)
12. [Configuration](#️-configuration)
13. [Running the Application](#-running-the-application)
14. [Usage](#-usage)
15. [Troubleshooting](#-usage)
17. [HyperParaMeters](#-intelligence-stack--hyperparameters)
16. [Team](#-team)

---

## 🎯 Introduction
**Sigma-AI** is a GenAI-enabled Integrated Platform Environment (IPE) application that empowers platform engineering teams by combining multiple tool capabilities into a **single intelligent interface**. It reduces context switching, streamlines operations, and introduces **Agentic AI capabilities** like:

- 🔁 Contextual understanding of incidents and system data  
- 💡 Intelligent recommendations based on historical and real-time context  
- 🧠 Root cause analysis using LLMs  
- ✍️ Automated description reframing for better incident clarity  
- 🔍 Extraction of key data points and correlations from change logs, incident history, and trace logs  

## 🎥 Demo
🔗 [Live Demo](#) (if applicable)  

## 🖥️ User-Interface Overview

![Sigma-AI UI Preview](artifacts/images/mermain-chart-min.png)

- 🧠 **Smart Issue Explorer**  
  Converts natural language issues into embeddings, finds similar incidents via FAISS (L2 distance), and generates GenAI-powered RCA and correlated CR suggestions.

- 🧾 **Incident Investigator**  
  Analyzes specific incidents to generate detailed RCA and suggest resolutions.

- 🧬 **TraceIQ**  
  GenAI-driven log summarization and troubleshooting based on trace IDs.

- 🌐 **NetViz Explorer**  
  Visualizes dependencies and suggests architectural insights powered by GenAI.

- 💬 **Agentic Chatbot: Dorra**  
  Assists with self-service RCA, logs, and CMDB queries.

---

## 🔄 Multi-Model Strategy – Smart Fallback Mechanism

Sigma-AI uses multiple models intelligently:

- ⚡ **Primary (OpenAI GPT-3.5)**: Detailed RCA, network insights, general queries.
- 🔄 **Fallback (LaMini-Flan-T5-783M)**: CPU-friendly text generation for RCA.
- 📘 **Summarization (facebook/bart-large-cnn)**: Document summarization.
- 📋 **Log Summarization (sshleifer/distilbart-cnn-12-6)**: Efficient log summarization.

---

## 📊 Sigma-AI Solution Architecture Overview

<p align="center">
  <img src="artifacts/images/image.png" alt="Sigma AI Architecture Diagram" width="500"/>
</p>

---

## 📐 Incident Similarity Search with L2 (Euclidean) Distance

Sigma-AI utilizes FAISS and L2 distance for efficient semantic similarity search.

$$
\text{distance} = \sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2 + \cdots + (x_n - y_n)^2}
$$

---

## 🧠 Intelligence Stack

- 🔡 **Embedding**: all-MiniLM-L6-v2 (SentenceTransformers)
- 🔍 **Vector Search**: FAISS  
- 🧠 **LLMs**:
  - OpenAI GPT-3.5 (Primary)
  - LaMini-Flan-T5-783M (Fallback)
  - facebook/bart-large-cnn (Document Summarization)
  - sshleifer/distilbart-cnn-12-6 (Log Summarization)

---

## 📂 Project Structure

```
.
📂 Sigma-AI
├── streamlit_app.py
├── app/
│   ├── data_loader.py
│   ├── vector_search.py
│   ├── model_runner.py
│   ├── change_checker.py
│   ├── log_checker.py
│   ├── chatbot.py
│   ├── network_viz.py
│   └── intelscope.py
├── data/
│   ├── incident_data.csv
│   ├── change.csv
│   ├── CMDB_Mapping.csv
│   ├── Logs_Lookup.csv
│   └── network_metadata.csv
└── requirements.txt
```

---

## 💻 Installation

Clone the repository:
```
git clone https://github.com/your-username/sigma-ai.git
cd sigma-ai
```

Create virtual environment:
- **Linux/macOS:**
```
python -m venv sigma_env
source sigma_env/bin/activate
```
- **Windows:**
```
python -m venv sigma_env
sigma_env\Scriptsctivate
```

Install dependencies:
```
pip install -r requirements.txt
```

---

## ⚙️ Configuration

Set your OpenAI API key:
```
export OPENAI_API_KEY=your-key
```

---

## 🚀 Running the Application

```
streamlit run streamlit_app.py
```

---

## 📖 Usage

1. Smart Issue Explorer: Natural language symptom analysis.
2. Incident Investigator: RCA generation for specific incidents.
3. TraceIQ: Log analysis and summarization.
4. NetViz Explorer: Dependency visualization.
5. Chatbot: Self-help and system queries.

---

## 🚀 Intelligence Stack & Hyperparameters

| Module/File          | Model(s)                                                  | Hyperparameters & Settings                                                                                 |
|----------------------|-----------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| `intelscope.py`      | Facebook BART Large CNN (`facebook/bart-large-cnn`)       | `max_length=200`, `min_length=50`, `do_sample=False`, Chunk size=1024 tokens, Step size=800 tokens        |
| `model_runner.py`    | Hugging Face LaMini-Flan-T5-783M (`MBZUAI/LaMini-Flan-T5-783M`) | `max_length=512`, `do_sample=True`, `temperature=0.7`, `top_p=0.9`, Input truncate=1100 tokens            |
| `model_runner.py`    | OpenAI GPT-3.5 Turbo (`gpt-3.5-turbo`)                    | `temperature=0.3` (RCA), `temperature=0.4` (Network queries)                                              |
| `vector_search.py`   | SentenceTransformer (`all-MiniLM-L6-v2`) + FAISS          | `top_k=3`, Vector Index: FAISS `IndexFlatL2` (squared Euclidean distance)                                 |


---

## ⚠️ Troubleshooting

- Verify OpenAI API key setup.
- Confirm all Hugging Face models installed.
- Check connectivity to log APIs.

---

## 👥 Team
- **Pradeep Malineni** - [GitHUb](https://github.com/malineni4u) | [LinkedIn](www.linkedin.com/in/pradeep-malineni)
- **Bhumika Peshwani** - [GitHUb](https://github.com/Bhumika158) | [LinkedIn](www.linkedin.com/in/bhumika158)

----
