#agents.ai_educator.task_manager.py
import logging

from server.task_manager import InMemoryTaskManager
from models.request import SendTaskRequest, SendTaskResponse
from models.task import Message, TaskStatus, TaskState, TextPart
from agents.ai_educator.agent import SimpleAIExplainer

logger = logging.getLogger(__name__)


class AIEducatorTaskManager(InMemoryTaskManager):
    """
    TaskManager for Simple AI Explainer:
    
    Handles educational interactions by:
    - Processing AI concept questions
    - Coordinating with AIXpert for technical knowledge
    - Providing structured Concept → Example → Conclusion explanations
    - Using analogies to make complex concepts simple
    """
    
    def __init__(self, agent: SimpleAIExplainer):
        """
        Initialize with Simple AI Explainer agent.
        
        Args:
            agent: The Simple AI Explainer instance
        """
        super().__init__()
        self.agent = agent

    def _get_user_text(self, request: SendTaskRequest) -> str:
        """Extract user text from the request."""
        return request.params.message.parts[0].text

    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        """
        Handle educational task with structured analogical explanation:
        
        1. Store the incoming AI concept question
        2. Process with Simple AI Explainer (may consult AIXpert)
        3. Return structured Concept → Example → Conclusion response
        """
        logger.info(f"Simple AI Explainer received concept question: {request.params.id}")

        task = await self.upsert_task(request.params)

        user_text = self._get_user_text(request)
        logger.info(f"Processing concept explanation request: '{user_text[:100]}...'")

        try:
            analogical_response = await self.agent.invoke(
                user_text,
                request.params.sessionId
            )
            logger.info(f"Generated analogical explanation: {len(analogical_response)} characters")
            
        except Exception as e:
            logger.error(f"Error generating analogical explanation: {e}")
            analogical_response = (
                "I'm having a small technical difficulty right now, but I'm still here to help explain AI concepts! "
                "Could you try asking your question again? I love breaking down complex AI topics into simple analogies! 🎓"
            )

        reply_message = Message(
            role="agent",
            parts=[TextPart(text=analogical_response)]
        )

        async with self.lock:
            task.status = TaskStatus(state=TaskState.COMPLETED)
            task.history.append(reply_message)

        return SendTaskResponse(id=request.id, result=task)