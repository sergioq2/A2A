#agents.aixpert_agent.__main__.py
from server.server import A2AServer

from models.agent import AgentCard, AgentCapabilities, AgentSkill

from agents.aixpert_agent.task_manager import AgentTaskManager
from agents.aixpert_agent.agent import AIXpertAgent

import click
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--host", default="localhost", help="Host to bind the server to")
@click.option("--port", default=10000, help="Port number for the server")
def main(host, port):

    capabilities = AgentCapabilities(streaming=False)

    skill = AgentSkill(
        id="ai_expert",
        name="AI Expert Assistant",
        description="Provides expert answers on AI topics including machine learning, deep learning, neural networks, and AI applications",
        tags=["ai", "machine learning", "deep learning", "neural networks", "artificial intelligence"],
        examples=["What is machine learning?", "Explain neural networks", "How does GPT work?", "What are the types of AI?"]
    )

    agent_card = AgentCard(
        name="AIXpertAgent",
        description="This agent provides expert knowledge and answers on artificial intelligence topics, machine learning, deep learning, and AI applications.",  # Description
        url=f"http://{host}:{port}/",
        version="1.0.0",
        defaultInputModes=AIXpertAgent.SUPPORTED_CONTENT_TYPES,
        defaultOutputModes=AIXpertAgent.SUPPORTED_CONTENT_TYPES,
        capabilities=capabilities,
        skills=[skill]
    )

    server = A2AServer(
        host=host,
        port=port,
        agent_card=agent_card,
        task_manager=AgentTaskManager(agent=AIXpertAgent())
    )

    server.start()

if __name__ == "__main__":
    main()