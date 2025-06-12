#agents.aixpert_agent.agent.py
import logging

from pygeai.chat.managers import ChatManager
from pygeai.core.models import ChatMessageList, ChatMessage, LlmSettings


logger = logging.getLogger(__name__)

class AIXpertAgent:
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self):
        """
        👷 Initialize the AIXpertAgent:
        - Creates the ChatManager for your existing AIXpert agent
        - Sets up LLM settings for consistent responses
        """
        self.agent_name = "AIXpert"
        self.chat_manager = ChatManager()
        
        self.llm_settings = LlmSettings(
            temperature=0.2,
            max_tokens=10000
        )
        
        logger.info(f"Initialized {self.agent_name} agent with pygeai ChatManager")

    async def invoke(self, query: str, session_id: str) -> str:
        """
        📥 Handle a user query and return a response string using your existing AIXpert agent.

        Args:
            query (str): What the user asked (e.g., "What is LoRA in fine tuning?")
            session_id (str): Helps group messages into a session (for future session management)

        Returns:
            str: Agent's reply (expert answer on AI topic from your AIXpert agent)
        """
        
        try:
            logger.info(f"Processing query with {self.agent_name}: {query}")
            
            messages = ChatMessageList(messages=[
                ChatMessage(role="user", content=query)
            ])
            
            response = self.chat_manager.chat_completion(
                model=f"saia:agent:{self.agent_name}",
                messages=messages,
                llm_settings=self.llm_settings
            )
            
            if hasattr(response, 'choices') and response.choices:
                answer = response.choices[0].message.content
                logger.info(f"Successfully received response from {self.agent_name}")
                return answer
            else:
                logger.error(f"Error in {self.agent_name} response: {response}")
                return f"I apologize, but I encountered an error while processing your AI question. Please try again."
                
        except Exception as e:
            logger.error(f"Exception in {self.agent_name} invoke: {str(e)}")
            return f"I apologize, but I encountered an error: {str(e)}. Please try again."


    async def stream(self, query: str, session_id: str):
        """
        🌀 Simulates a "streaming" agent that returns AI expert responses.
        This uses your existing AIXpert agent and yields the response.

        Yields:
            dict: Response payload with the AI expert answer
        """
        try:
            response_text = await self.invoke(query, session_id)
            
            yield {
                "is_task_complete": True,
                "content": response_text
            }
        except Exception as e:
            yield {
                "is_task_complete": True,
                "content": f"Error processing your AI question: {str(e)}"
            }