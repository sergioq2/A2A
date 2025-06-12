#agents.ai_educator.__main__.py
import logging
import click

from server.server import A2AServer
from models.agent import AgentCard, AgentCapabilities, AgentSkill
from agents.ai_educator.task_manager import AIEducatorTaskManager
from agents.ai_educator.agent import SimpleAIExplainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.command()
@click.option("--host", default="localhost", help="Host to bind AI Educator server to")
@click.option("--port", default=10001, help="Port for AI Educator server")
@click.option("--aixpert-url", default="http://localhost:10000", help="URL of AIXpert agent")
def main(host: str, port: int, aixpert_url: str):
    """
    Launches the Simple AI Explainer A2A server.
    
    This agent specializes in making complex AI concepts simple and memorable through:
    - Everyday analogies and metaphors
    - Structured Concept → Example → Conclusion format  
    - Casual, friendly language
    - Direct consultation with AIXpert for technical accuracy
    """
    print(f"\n🎓 Starting Simple AI Explainer on http://{host}:{port}/")
    print(f"Connected to AIXpert at {aixpert_url}")
    print("📚 Specializes in: Concept → Example → Conclusion structure with analogies\n")

    capabilities = AgentCapabilities(streaming=False)

    skill = AgentSkill(
        id="simple_ai_teaching",
        name="Simple AI Explainer",
        description="Makes complex AI concepts incredibly simple using analogies and structured explanations (Concept → Example → Conclusion)",
        tags=["education", "ai", "analogies", "simple", "explanation", "concepts", "beginner-friendly"],
        examples=[
            "What is machine learning?",
            "Explain neural networks in simple terms",
            "What are the ARIME models ?",
            "What are transformers in AI?",
            "Break down deep learning for me",
            "Explain fine-tuning like I'm 10",
            "What's the difference between AI and ML?",
            "How do large language models work?"
        ]
    )

    agent_card = AgentCard(
        name="SimpleAIExplainer",
        description=(
            "An AI educator specialized in making complex AI concepts incredibly simple and memorable "
            "through everyday analogies and structured explanations. Uses a consistent "
            "Concept → Example → Conclusion format with casual, friendly language that anyone can understand."
        ),
        url=f"http://{host}:{port}/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=capabilities,
        skills=[skill]
    )

    simple_ai_explainer = SimpleAIExplainer(aixpert_url=aixpert_url)
    task_manager = AIEducatorTaskManager(agent=simple_ai_explainer)

    server = A2AServer(
        host=host,
        port=port,
        agent_card=agent_card,
        task_manager=task_manager
    )
    server.start()

if __name__ == "__main__":
    main()