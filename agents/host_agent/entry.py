#agents.host_agent.entry.py
import asyncio
import logging
import click

from utilities.discovery import DiscoveryClient
from server.server import A2AServer
from models.agent import AgentCard, AgentCapabilities, AgentSkill
from agents.host_agent.orchestrator import (
    OrchestratorAgent,
    OrchestratorTaskManager
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.command()
@click.option(
    "--host", default="localhost",
    help="Host to bind the OrchestratorAgent server to"
)
@click.option(
    "--port", default=10002,
    help="Port for the OrchestratorAgent server"
)
@click.option(
    "--registry",
    default=None,
    help=(
        "Path to JSON file listing child-agent URLs. "
        "Defaults to utilities/agent_registry.json"
    )
)
def main(host: str, port: int, registry: str):
    """
    Entry point to start the OrchestratorAgent A2A server.

    Steps performed:
    1. Load child-agent URLs from the registry JSON file.
    2. Fetch each agent's metadata via /.well-known/agent.json.
    3. Instantiate an OrchestratorAgent with discovered AgentCards.
    4. Wrap it in an OrchestratorTaskManager for JSON-RPC handling.
    5. Launch the A2AServer to listen for incoming tasks.
    """
    discovery = DiscoveryClient(registry_file=registry)
    agent_cards = asyncio.run(discovery.list_agent_cards())

    if not agent_cards:
        logger.warning(
            "No agents found in registry – the orchestrator will have nothing to call"
        )

    capabilities = AgentCapabilities(streaming=False)
    skill = AgentSkill(
        id="orchestrate",
        name="Orchestrate Tasks",
        description=(
            "Routes user requests to the appropriate child agent: "
            "Simple  SimpleAIExplainer for easy-to-understand AI explanations with analogies"
        ),
        tags=["routing", "orchestration", "simple-ai", "analogies", "ai-educator"],
        examples=[
            "What is machine learning?",
            "Explain neural networks simply",
            "Use an analogy for transformers",
            "Break down PEFT for me",
            "What are LLMs in simple terms?",
            "How does fine-tuning work?",
            "Explain deep learning like I'm 10",
            "Advanced AI research topics",
            "What is LoRA in fine tuning"
        ]
    )
    orchestrator_card = AgentCard(
        name="OrchestratorAgent",
        description=(
            "Intelligent task router that delegates user requests to specialized agents: "
            "SimpleAIExplainer for easy-to-understand AI concepts using analogies and structured explanations, "
        ),
        url=f"http://{host}:{port}/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=capabilities,
        skills=[skill]
    )

    orchestrator = OrchestratorAgent(agent_cards=agent_cards)
    task_manager = OrchestratorTaskManager(agent=orchestrator)

    server = A2AServer(
        host=host,
        port=port,
        agent_card=orchestrator_card,
        task_manager=task_manager
    )

    logger.info(
        f"Starting OrchestratorAgent with {len(agent_cards)} discovered agents: "
        f"AI Educator that teach about specific AI Topic"
    )
    server.start()

if __name__ == "__main__":
    main()