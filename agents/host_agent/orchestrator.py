#agents.host_agent.orchestrator.py
import os 
import uuid
import logging
from dotenv import load_dotenv


load_dotenv()

# -----------------------------------------------------------------------------
# Google ADK / Gemini imports
# -----------------------------------------------------------------------------
from google.adk.agents.llm_agent import LlmAgent
# LlmAgent: core class to define a Gemini-powered AI agent

from google.adk.sessions import InMemorySessionService
# InMemorySessionService: stores session state in memory (for simple demos)

from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
# InMemoryMemoryService: optional conversation memory stored in RAM

from google.adk.artifacts import InMemoryArtifactService
# InMemoryArtifactService: handles file/blob artifacts (unused here)

from google.adk.runners import Runner
# Runner: orchestrates agent, sessions, memory, and tool invocation

from google.adk.agents.readonly_context import ReadonlyContext
# ReadonlyContext: passed to system prompt function to read context

from google.adk.tools.tool_context import ToolContext
# ToolContext: passed to tool functions for state and actions

from google.genai import types           
# types.Content & types.Part: used to wrap user messages for the LLM

# -----------------------------------------------------------------------------
# A2A server-side infrastructure
# -----------------------------------------------------------------------------
from server.task_manager import InMemoryTaskManager
# InMemoryTaskManager: base class providing in-memory task storage and locking

from models.request import SendTaskRequest, SendTaskResponse
# Data models for incoming task requests and outgoing responses

from models.task import Message, TaskStatus, TaskState, TextPart
# Message: encapsulates role+parts; TaskStatus/State: status enums; TextPart: text payload

# -----------------------------------------------------------------------------
# Connector to child A2A agents
# -----------------------------------------------------------------------------
from agents.host_agent.agent_connect import AgentConnector
# AgentConnector: lightweight wrapper around A2AClient to call other agents

from models.agent import AgentCard
# AgentCard: metadata structure for agent discovery results

# Set up module-level logger for debug/info messages
logger = logging.getLogger(__name__)


class OrchestratorAgent:
    """
    Uses a Gemini LLM to route incoming user queries,
    calling out to any discovered child A2A agents via tools.
    Specialized for SimpleAIExplainer (simple analogical AI education) and AIXpertAgent (technical AI expertise).
    """

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self, agent_cards: list[AgentCard]):
        # Build one AgentConnector per discovered AgentCard
        # agent_cards is a list of AgentCard objects returned by discovery
        self.connectors = {
            card.name: AgentConnector(card.name, card.url)
            for card in agent_cards
        }

        # Log discovered agents for debugging
        agent_names = list(self.connectors.keys())
        logger.info(f"OrchestratorAgent initialized with agents: {agent_names}")

        # Build the internal LLM agent with our custom tools and instructions
        self._agent = self._build_agent()

        # Static user ID for session tracking across calls
        self._user_id = "orchestrator_user"

        # Runner wires up sessions, memory, artifacts, and handles agent.run()
        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

    def _build_agent(self) -> LlmAgent:
        """
        Construct the Gemini-based LlmAgent with:
        - Model name
        - Agent name/description
        - System instruction callback
        - Available tool functions
        """
        return LlmAgent(
            model="gemini-1.5-flash-latest",    # Specify Gemini model version
            name="orchestrator_agent",          # Human identifier for this agent
            description="Intelligent router for AI Educator that tech about AI Topics",
            instruction=self._root_instruction,  # Function providing system prompt text
            tools=[
                self._list_agents,               # Tool 1: list available child agents
                self._delegate_task             # Tool 2: call a child agent
            ],
        )

    def _root_instruction(self, context: ReadonlyContext) -> str:
        """
        Enhanced system prompt function: returns instruction text for the LLM,
        including routing guidelines for SimpleAIExplainer vs AIXpertAgent.
        """
        # Build a bullet-list of agent names
        agent_list = "\n".join(f"- {name}" for name in self.connectors)
        
        return (
            "You are an intelligent orchestrator that routes user queries to specialized agents.\n\n"
            "AVAILABLE TOOLS:\n"
            "1) list_agents() -> list available child agents\n"
            "2) delegate_task(agent_name, message) -> call that agent with the user's message\n\n"
            
            "ROUTING GUIDELINES:\n"
            "- SimpleAIExplainer: Use for AI education with simple explanations, analogies, and beginner-friendly learning\n"
            "  Examples: 'What is ML?', 'Explain transformers simply', 'Break down PEFT for me', 'Use an analogy for neural networks'\n\n"
            
            "- ai_educator: Use for advanced technical AI questions that need deep expertise and detailed technical information\n"
            "  Examples: 'Latest research in transformers', 'Technical implementation details', 'Advanced optimization techniques'\n\n"
            
            "INSTRUCTIONS:\n"
            "- Always use delegate_task() to forward the user's EXACT message to the appropriate agent\n"
            "- Do not modify or rephrase the user's question\n"
            "- Choose the most appropriate agent based on the query intent\n"
            "- For technical AI expertise: use ai_educator\n"
            
            f"AVAILABLE AGENTS:\n{agent_list}\n\n"
            "Always use the tools provided. Do not hallucinate responses."
        )

    def _list_agents(self) -> list[str]:
        """
        Tool function: returns the list of child-agent names currently registered.
        Called by the LLM when it wants to discover available agents.
        """
        agents = list(self.connectors.keys())
        logger.debug(f"Listed agents: {agents}")
        return agents

    async def _delegate_task(
        self,
        agent_name: str,
        message: str,
        tool_context: ToolContext
    ) -> str:
        """
        Tool function: forwards the `message` to the specified child agent
        (via its AgentConnector), waits for the response, and returns the
        text of the last reply.
        """
        # Validate agent_name exists
        if agent_name not in self.connectors:
            available_agents = list(self.connectors.keys())
            raise ValueError(f"Unknown agent: {agent_name}. Available agents: {available_agents}")
        
        connector = self.connectors[agent_name]
        logger.info(f"Delegating task to {agent_name}: '{message[:50]}...'")

        # Ensure session_id persists across tool calls via tool_context.state
        state = tool_context.state
        if "session_id" not in state:
            state["session_id"] = str(uuid.uuid4())
        session_id = state["session_id"]

        # Delegate task asynchronously and await Task result
        child_task = await connector.send_task(message, session_id)
        logger.info(f"Received response from {agent_name}")

        # Extract text from the last history entry if available
        if child_task.history and len(child_task.history) > 1:
            response = child_task.history[-1].parts[0].text
            logger.debug(f"Response from {agent_name}: '{response[:100]}...'")
            return response
        
        logger.warning(f"No response received from {agent_name}")
        return ""

    async def invoke(self, query: str, session_id: str) -> str:

        logger.info(f"OrchestratorAgent processing query: '{query[:50]}...'")
        
        if "SimpleAIExplainer" in self.connectors:
            logger.info("🚀 Delegating directly to SimpleAIExplainer")
            connector = self.connectors["SimpleAIExplainer"]
            
            try:
                child_task = await connector.send_task(query, session_id)
                
                if child_task.history and len(child_task.history) > 1:
                    response = child_task.history[-1].parts[0].text
                    logger.info(f"SimpleAIExplainer response received: {len(response)} chars")
                    return response
                else:
                    logger.warning("No response from SimpleAIExplainer")
                    return "I couldn't get a response from the AI educator right now."
                    
            except Exception as e:
                logger.error(f"Error delegating to SimpleAIExplainer: {e}")
                return f"I'm having trouble connecting to the AI educator: {str(e)}"
        
        else:
            logger.error("SimpleAIExplainer not found in connectors")
            available = list(self.connectors.keys())
            return f"SimpleAIExplainer not available. Available agents: {available}"


class OrchestratorTaskManager(InMemoryTaskManager):
    """
    🪄 TaskManager wrapper: exposes OrchestratorAgent.invoke() over the
    A2A JSON-RPC `tasks/send` endpoint, handling in-memory storage and
    response formatting.
    """
    def __init__(self, agent: OrchestratorAgent):
        super().__init__()       # Initialize base in-memory storage
        self.agent = agent       # Store our orchestrator logic

    def _get_user_text(self, request: SendTaskRequest) -> str:
        """
        Helper: extract the user's raw input text from the request object.
        """
        return request.params.message.parts[0].text

    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        """
        Called by the A2A server when a new task arrives:
        1. Store the incoming user message
        2. Invoke the OrchestratorAgent to get a response
        3. Append response to history, mark completed
        4. Return a SendTaskResponse with the full Task
        """
        logger.info(f"OrchestratorTaskManager received task {request.params.id}")

        # Step 1: save the initial message
        task = await self.upsert_task(request.params)

        # Step 2: run orchestration logic
        user_text = self._get_user_text(request)
        response_text = await self.agent.invoke(user_text, request.params.sessionId)

        # Step 3: wrap the LLM output into a Message
        reply = Message(role="agent", parts=[TextPart(text=response_text)])
        async with self.lock:
            task.status = TaskStatus(state=TaskState.COMPLETED)
            task.history.append(reply)

        # Step 4: return structured response
        return SendTaskResponse(id=request.id, result=task)