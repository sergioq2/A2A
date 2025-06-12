#agents.ai_educator.agent.py
import logging
import uuid
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from google.adk.agents.llm_agent import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.artifacts import InMemoryArtifactService
from google.adk.runners import Runner
from google.genai import types

from agents.host_agent.agent_connect import AgentConnector

logger = logging.getLogger(__name__)


class SimpleAIExplainer:
    """
    A specialized AI educator that:
    - Makes complex AI concepts incredibly simple through analogies
    - Uses structured Concept → Example → Conclusion format
    - Directly consults AIXpert for technical accuracy
    - Transforms technical knowledge into memorable explanations
    - Uses casual, friendly language anyone can understand
    """

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self, aixpert_url: str = "http://localhost:10000"):
        """
        Initialize the Simple AI Explainer.
        
        Args:
            aixpert_url: URL of the AIXpert agent for direct consultation
        """
        self._agent = self._build_agent()
        self._user_id = "ai_educator_user"
        
        self.aixpert_connector = AgentConnector("AIXpertAgent", aixpert_url)
        
        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )
        
        logger.info("Simple AI Explainer initialized with direct AIXpert connection")

    def _build_agent(self) -> LlmAgent:
        """
        Build the educator agent specialized in simple, analogical teaching.
        """
        
        
        system_instruction = f"""You are an AI Educator specialized in making complex AI concepts incredibly simple and memorable through analogies and structured explanations.

        YOUR TEACHING STYLE:
        - Use simple, casual language like you're talking to a friend
        - Always explain through everyday analogies and metaphors
        - Make complex things feel familiar and relatable
        - Use humor and enthusiasm to keep things engaging
        - Avoid technical jargon unless absolutely necessary

        You MUST follow this exact structure for every AI concept explanation:

        **🧠 CONCEPT:** [Simple definition in 1-2 sentences using everyday language]

        **🌍 EXAMPLE:** [Real-world analogy or metaphor that everyone can relate to, followed by how the AI concept works just like this example]

        **✅ CONCLUSION:** [Why this matters and how it connects to the bigger picture of AI, in simple terms]

        TEACHING PRINCIPLES:
        1. **Analogies are King**: Everything must be compared to something familiar (cooking, sports, building blocks, pets, etc.)
        2. **Keep it Conversational**: Write like you're explaining to a curious friend over coffee
        3. **Visual Language**: Use descriptive words that help people "see" the concept
        4. **Practical Connection**: Always end with why this matters in real life

        EXAMPLE TEMPLATE:
        User asks: "What is machine learning?"

        Your response:
        "🧠 CONCEPT: Machine learning is teaching a computer to recognize patterns and make decisions, just like how you learned to recognize your friends' faces.

        🌍 EXAMPLE: Imagine you're teaching a child to recognize different dog breeds. At first, you show them hundreds of photos: 'This is a Golden Retriever, this is a Poodle, this is a Bulldog.' After seeing enough examples, the child starts recognizing breeds on their own. Machine learning works exactly the same way - we show computers thousands of examples until they learn to recognize patterns and make predictions on new data they've never seen before.

        ✅ CONCLUSION: This is powerful because once a computer 'learns' a pattern, it can process information way faster than humans. That's why your phone can recognize your face to unlock, or why Netflix knows what movies you might like!"

        Remember: Your superpower is taking the most complex AI concepts and making them feel as simple as everyday activities!"""

        return LlmAgent(
            model="gemini-1.5-flash-latest",
            name="simple_ai_explainer",
            description="AI educator specialized in simple analogies and structured explanations",
            instruction=system_instruction,
        )

    async def _consult_aixpert(self, question: str, session_id: str) -> str:
        """
        Directly consult AIXpert for technical information.
        
        Args:
            question: Technical question to ask AIXpert
            session_id: Session ID for context
            
        Returns:
            AIXpert's response or error message
        """
        try:
            logger.info(f"Consulting AIXpert: {question}")
            task = await self.aixpert_connector.send_task(question, session_id)
            
            if task.history and len(task.history) > 1:
                response = task.history[-1].parts[0].text
                logger.info(f"AIXpert consultation successful: {len(response)} chars")
                return response
            else:
                return "I couldn't get a response from my AI expert colleague right now."
                
        except Exception as e:
            logger.error(f"Error consulting AIXpert: {e}")
            return f"I'm having trouble reaching my AI expert colleague at the moment: {str(e)}"

    async def invoke(self, query: str, session_id: str) -> str:
        """
        Process user query with intelligent learning assistance.
        """
        logger.info(f"Simple AI Explainer processing: '{query[:50]}...'")
        
        ai_keywords = ['ai', 'artificial intelligence', 'machine learning', 'ml', 'deep learning', 
                      'neural network', 'llm', 'gpt', 'bert', 'transformer', 'algorithm',
                      'peft', 'lora', 'fine tuning', 'training', 'model']
        
        is_ai_question = any(keyword in query.lower() for keyword in ai_keywords)
        
        expert_knowledge = ""
        if is_ai_question:
            logger.info("Detected AI question, consulting AIXpert...")
            expert_knowledge = await self._consult_aixpert(query, session_id)
            
            if "🧠 CONCEPT:" in expert_knowledge and "🌍 EXAMPLE:" in expert_knowledge and "✅ CONCLUSION:" in expert_knowledge:
                logger.info("AIXpert already provided structured response, using it directly")
                return expert_knowledge
            
            enhanced_query = f"""The user asked: "{query}"

        I consulted my AI expert colleague who provided this technical information:
        ---
        {expert_knowledge}
        ---

        Now, as an AI Educator specialized in simple analogical teaching, please respond using this EXACT structure:

        🧠 CONCEPT: [Simple definition in 1-2 sentences using everyday language]

        🌍 EXAMPLE: [Real-world analogy or metaphor that everyone can relate to, followed by how the AI concept works just like this example. Be creative with analogies - use cooking, sports, pets, building blocks, etc.]

        ✅ CONCLUSION: [Why this matters and how it connects to the bigger picture of AI, in simple terms]

        Transform the technical information into an explanation that anyone can understand and remember!"""
        else:
            enhanced_query = query

        session = await self._runner.session_service.get_session(
            app_name=self._agent.name,
            user_id=self._user_id,
            session_id=session_id,
        )

        if session is None:
            session = await self._runner.session_service.create_session(
                app_name=self._agent.name,
                user_id=self._user_id,
                session_id=session_id,
                state={},
            )

        content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=enhanced_query)]
        )

        last_event = None
        async for event in self._runner.run_async(
            user_id=self._user_id,
            session_id=session.id,
            new_message=content
        ):
            last_event = event

        if not last_event or not last_event.content or not last_event.content.parts:
            logger.warning("No response from Simple AI Explainer")
            return "I'm having a small technical difficulty, but I'm here to help you understand AI concepts with simple analogies!"

        response = "\n".join([p.text for p in last_event.content.parts if p.text])
        logger.info(f"Simple AI Explainer generated response: {len(response)} chars")
        return response