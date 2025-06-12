#agents.aixpert_agent.task_manager.py
import logging

from server.task_manager import InMemoryTaskManager

from agents.aixpert_agent.agent import AIXpertAgent

from models.request import SendTaskRequest, SendTaskResponse
from models.task import Message, Task, TextPart, TaskStatus, TaskState


logger = logging.getLogger(__name__)


class AgentTaskManager(InMemoryTaskManager):
    """
    This class connects your existing AIXpert agent (using pygeai) to the task system.

    - It "inherits" all the logic from InMemoryTaskManager
    - It overrides the part where we handle a new task (on_send_task)
    - It uses your existing AIXpert agent to generate expert AI responses
    """

    def __init__(self, agent: AIXpertAgent):
        super().__init__()
        self.agent = agent

    def _get_user_query(self, request: SendTaskRequest) -> str:
        """
        Get the user's text input from the request object.

        Example: If the user asks "What is deep learning?", we pull that string out.

        Args:
            request: A SendTaskRequest object

        Returns:
            str: The actual text the user asked
        """
        return request.params.message.parts[0].text

    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        """
        This is the heart of the task manager.

        It does the following:
        1. Save the task into memory (or update it)
        2. Ask the AIXpert agent for an expert AI response
        3. Format that reply as a message
        4. Save the agent's reply into the task history
        5. Return the updated task to the caller
        """

        logger.info(f"Processing new AI query: {request.params.id}")

        task = await self.upsert_task(request.params)

        query = self._get_user_query(request)

        result_text = await self.agent.invoke(query, request.params.sessionId)

        agent_message = Message(
            role="agent",
            parts=[TextPart(text=result_text)]
        )

        async with self.lock:
            task.status = TaskStatus(state=TaskState.COMPLETED)
            task.history.append(agent_message)

        return SendTaskResponse(id=request.id, result=task)