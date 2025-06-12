# 🤖 Globant Agents Multi-Agent Demo – A2A with Google ADK

Welcome to Multi agent A2A GEAI integration with other frameworks — a minimal Agent2Agent (A2A) implementation using Google's [Agent Development Kit (ADK)] and Globant Agents python SDK(https://github.com/google/agent-development-kit).

This example demonstrates how to build, serve, and interact with three A2A agents:
1. **AIEducator** – explain an AI topic in a very simple way.
2. **AIXpert** – Uses an agentic RAG from GEAI to fetch information from both KB and web.
3. **OrchestratorAgent** – routes requests to the appropriate child agent.

All of them work together seamlessly via A2A discovery and JSON-RPC.

---

## 📦 Project Structure

```bash
version_3_multi_agent/
├── .env  
├── pyproject.toml
├── README.md
├── app/
│   └── cmd/
│       └── cmd.py 
├── agents/
│   ├── ai_educator/
│   │   ├── __main__.py
│   │   ├── agent.py
│   │   └── task_manager.py
│   ├── aixpert_agent/
│   │   ├── __main__.py
│   │   ├── agent.py
│   │   └── task_manager.py
│   └── host_agent/
│       ├── entry.py
│       ├── orchestrator.py
│       └── agent_connect.py
├── server/
│   ├── server.py
│   └── task_manager.py
└── utilities/
    ├── discovery.py
    └── agent_registry.json
```

---

## 🛠️ Setup

1. **Clone & navigate**

    ```bash
    git clone https://github.com/sergioq2/a2a.git
    cd a2a
    ```

2. **Create & activate a venv**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. **Install dependencies**

    Using [`uv`](https://github.com/astral-sh/uv):

    ```bash
    uv pip install .
    ```

    Or with pip directly:

    ```bash
    pip install .
    ```

4. **Set your API key**

    Create `.env` at the project root:
    ```bash
    echo "GOOGLE_API_KEY=your_api_key_here" > .env
    ```

---

## 🎬 Demo Walkthrough

**Start the ai educator**
```bash
python3 -m agents.ai_educator \
  --host localhost --port 10000
```

**Start the aixpert_agent**
```bash
python3 -m agents.aixpert_agent \
  --host localhost --port 10001
```

**Start the Orchestrator (Host) Agent**
```bash
python3 -m agents.host_agent.entry \
  --host localhost --port 10002
```

**Launch the CLI (cmd.py)**
```bash
python3 -m app.cmd.cmd --agent http://localhost:10002
```

**Try it out!**
```bash
> What is LoRA in fine tuning ?

---

## 🔍 How It Works

1. **Discovery**: OrchestratorAgent reads `utilities/agent_registry.json`, fetches each agent’s `/​.well-known/agent.json`.
2. **Routing**: Based on intent, the Orchestrator’s LLM calls its tools:
   - `list_agents()` → lists child-agent names
   - `delegate_task(agent_name, message)` → forwards tasks
3. **Child Agents**:
   - AI Educator Agent teach a topic in a very simple an structured way.
   - AIXpert Agent is an agentic RAG that search the topic asked in a KB and the web.
4. **JSON-RPC**: All communication uses A2A JSON-RPC 2.0 over HTTP via Starlette & Uvicorn.

---

## 📖 Learn More

- A2A GitHub: https://github.com/google/A2A  
- Google ADK: https://github.com/google/agent-development-kit