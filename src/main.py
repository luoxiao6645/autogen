import asyncio
import logging
import os
from typing import Callable, Optional, Dict, List, Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from autogen_agentchat.agents import ConversableAgent, AssistantAgent, UserProxyAgent
from autogen_core.models import ModelClient
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.memory.chromadb import ChromaDBVectorMemory, ChromaDBVectorMemoryConfig
from autogen_ext.experimental.task_centric_memory import MemoryController, Teachability

from src.agents.interaction_agent import UserInteractionAgent
from src.agents.orchestrator_agent import OrchestratorAgent
from src.agents.retrieval_agent import InformationRetrievalAgent
from src.agents.task_agent import TaskExecutionAgent
from src.agents.learning_agent import LearningMemoryAgent

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

RAG_PERSISTENCE_DIR = "./chroma_db_store"
RAG_COLLECTION_NAME = "personal_agent_docs"
global_vector_db: Optional[ChromaDBVectorMemory] = None

LEARNING_MEMORY_DIR = "./learning_memory_store"
LEARNING_MEMORY_FILE = os.path.join(LEARNING_MEMORY_DIR, "personal_agent_memory.json")
global_memory_controller: Optional[MemoryController] = None
global_teachability: Optional[Teachability] = None

@app.on_event("startup")
async def startup_event():
    global global_vector_db, global_memory_controller, global_teachability
    logger.info(f"FastAPI startup: Initializing ChromaDB for RAG: '{RAG_COLLECTION_NAME}' from '{RAG_PERSISTENCE_DIR}'...")
    if not os.path.exists(RAG_PERSISTENCE_DIR):
        logger.warning(f"ChromaDB persistence directory '{RAG_PERSISTENCE_DIR}' not found.")
    rag_chroma_config = ChromaDBVectorMemoryConfig(collection_name=RAG_COLLECTION_NAME, persistence_path=RAG_PERSISTENCE_DIR)
    try:
        global_vector_db = ChromaDBVectorMemory(config=rag_chroma_config)
        logger.info("ChromaDBVectorMemory for RAG initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDBVectorMemory for RAG: {e}", exc_info=True)

    logger.info(f"FastAPI startup: Initializing Learning Memory from '{LEARNING_MEMORY_FILE}'...")
    os.makedirs(LEARNING_MEMORY_DIR, exist_ok=True)
    api_key_for_memory = os.getenv("OPENAI_API_KEY")
    memory_llm_client = OpenAIChatCompletionClient(model="gpt-3.5-turbo", api_key=api_key_for_memory) if api_key_for_memory else None
    if not memory_llm_client: logger.warning("API key for MemoryController's LLM not found. Some memory operations might be limited.")

    try:
        global_memory_controller = MemoryController(
            path_to_memory_file=LEARNING_MEMORY_FILE, persist_memory=True, reset=False, client=memory_llm_client)
        global_teachability = Teachability(memory_controller=global_memory_controller, reset_memory=False)
        logger.info("MemoryController and Teachability for Learning Memory initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize MemoryController/Teachability: {e}", exc_info=True)

class AgentSession:
    def __init__(self,
                 shared_vector_db: Optional[ChromaDBVectorMemory],
                 shared_teachability: Optional[Teachability],
                 model_client: ModelClient):
        self.user_interaction_agent: Optional[UserInteractionAgent] = None
        self.orchestrator: Optional[OrchestratorAgent] = None
        self.retrieval_agent: Optional[InformationRetrievalAgent] = None
        self.task_agent: Optional[TaskExecutionAgent] = None
        self.learning_agent: Optional[LearningMemoryAgent] = None
        self.shared_vector_db = shared_vector_db
        self.shared_teachability = shared_teachability
        self.model_client = model_client
        self.chat_initiated = False
        self.input_future: Optional[asyncio.Future] = None
        self.client_websocket: Optional[WebSocket] = None
        self.current_orchestrator_chat_task: Optional[asyncio.Task] = None


    async def initialize_agents(self):
        if self.shared_vector_db:
            self.retrieval_agent = InformationRetrievalAgent(
                name="RetrievalExpert", model_client=self.model_client, vector_memory=self.shared_vector_db)
            logger.info("InformationRetrievalAgent initialized.")

        self.task_agent = TaskExecutionAgent(name="TaskExecutor", model_client=self.model_client)
        logger.info("TaskExecutionAgent initialized.")

        if self.shared_teachability:
            self.learning_agent = LearningMemoryAgent(
                name="Learner", model_client=self.model_client, teachability=self.shared_teachability)
            logger.info("LearningMemoryAgent initialized.")

        self.orchestrator = OrchestratorAgent(name="Orchestrator", model_client=self.model_client)

        tools_to_register_on_orchestrator = []
        if self.retrieval_agent:
            tools_to_register_on_orchestrator.append(self.retrieval_agent.answer_from_knowledge_base)
        if self.task_agent:
            tools_to_register_on_orchestrator.append(self.task_agent.send_email)
        if self.learning_agent: # Register Learner's recall tool with Orchestrator
            tools_to_register_on_orchestrator.append(self.learning_agent.recall_learned_information)

        if tools_to_register_on_orchestrator:
            self.orchestrator.register_tools(tools=tools_to_register_on_orchestrator)
            logger.info(f"Registered {len(tools_to_register_on_orchestrator)} tools with Orchestrator.")

        async def get_input_from_websocket(prompt: str) -> str:
            logger.info(f"Agent is requesting input: {prompt}")
            if self.client_websocket:
                await self.client_websocket.send_json({"role": "system", "content": prompt, "type": "input_request"})
            self.input_future = asyncio.get_event_loop().create_future()
            try: return await asyncio.wait_for(self.input_future, timeout=300)
            except asyncio.TimeoutError: logger.warning("Timeout waiting for user input."); return "No input."

        self.user_interaction_agent = UserInteractionAgent(
            name="UserWebAppProxy", human_input_mode="ALWAYS", input_func=get_input_from_websocket)
        logger.info("UserInteractionAgent initialized.")
        return True

    async def handle_message(self, message_content: str, websocket: WebSocket):
        if not self.user_interaction_agent or not self.orchestrator or not self.learning_agent:
            logger.error("Core agents (User, Orchestrator, or Learner) not initialized.")
            await websocket.send_json({"role": "error", "content": "A core agent is not initialized on server."})
            return

        if self.input_future and not self.input_future.done():
            self.input_future.set_result(message_content)
        elif not self.chat_initiated or (self.current_orchestrator_chat_task and self.current_orchestrator_chat_task.done()):
            self.chat_initiated = True
            logger.info(f"Orchestrator handling message: {message_content}")

            # Main reply function for orchestrator to send messages to WebSocket
            async def ws_send_from_orchestrator(messages: List[Dict[str, Any]], sender: ConversableAgent, recipient: ConversableAgent) -> List[Dict[str, Any]]:
                last_message = messages[-1]
                if recipient.name == self.user_interaction_agent.name and sender.name == self.orchestrator.name:
                    content_to_send = last_message.get("content", "")
                    logger.info(f"Orchestrator sending to UI: {content_to_send[:200]}...") # Log snippet

                    # Detect if Orchestrator wants to teach Learner (based on its system prompt)
                    # This is a simplified detection. More robust would be a specific tool call.
                    teach_keywords = ["Okay, I will tell the Learner to remember:", "Okay, I'll pass this to Learner:"]
                    is_teaching_intent = any(keyword.lower() in content_to_send.lower() for keyword in teach_keywords)

                    if is_teaching_intent and self.learning_agent:
                        # Extract the part to teach (this is heuristic)
                        actual_info_to_teach = content_to_send
                        for keyword in teach_keywords: # Try to strip the keyword prefix
                            if keyword.lower() in actual_info_to_teach.lower():
                                actual_info_to_teach = actual_info_to_teach[actual_info_to_teach.lower().find(keyword.lower()) + len(keyword):].strip()
                                break

                        logger.info(f"Orchestrator intends to teach Learner: '{actual_info_to_teach}'")
                        # Orchestrator initiates a sub-chat with Learner to teach
                        # This sub-chat is silent to the main UI for now.
                        # The Learner's Teachability will pick up from this message.
                        await self.orchestrator.a_initiate_chat(
                            recipient=self.learning_agent,
                            message=f"Please learn this: {actual_info_to_teach}", # Orchestrator explicitly tells Learner
                            clear_history=True, # Fresh context for this teaching interaction
                            max_turns=1 # Orchestrator tells, Learner (via Teachability) learns and might acknowledge.
                        )
                        # After teaching, Orchestrator might need to send a different confirmation to user.
                        # For now, we just send the original orchestrator message that indicated intent.
                        # A more advanced flow would have Orchestrator wait for Learner's ack, then confirm to user.

                    await websocket.send_json({"role": "assistant", "content": content_to_send})
                return messages

            # Clear previous reply functions to avoid duplicates if session is long-lived for one user.
            self.orchestrator.clear_reply_functions()
            self.orchestrator.register_reply_function(
                trigger=self.user_interaction_agent, reply_function=ws_send_from_orchestrator)

            async def initiate_and_run_chat_with_orchestrator():
                await self.user_interaction_agent.a_initiate_chat(
                    recipient=self.orchestrator, clear_history=False, message=message_content) # Keep orchestrator history for context
                logger.info("Main chat turn with Orchestrator complete or awaiting input.")

            self.current_orchestrator_chat_task = asyncio.create_task(initiate_and_run_chat_with_orchestrator())
        else:
            logger.warning(f"Unexpected message: {message_content} (chat_initiated: {self.chat_initiated}, task_done: {self.current_orchestrator_chat_task.done() if self.current_orchestrator_chat_task else 'N/A'})")


@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept(); logger.info("WebSocket connection established.")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found."); await websocket.send_json({"role": "error", "content": "OpenAI API key not configured."}); await websocket.close(); return
    if global_vector_db is None:
        logger.error("RAG Vector DB not available."); await websocket.send_json({"role": "error", "content": "Knowledge base (RAG DB) not available."}); await websocket.close(); return
    if global_teachability is None or global_memory_controller is None:
        logger.error("Learning Memory not available."); await websocket.send_json({"role": "error", "content": "Learning memory system not available."}); await websocket.close(); return

    model_client = OpenAIChatCompletionClient(model="gpt-4o", api_key=api_key)
    agent_session = AgentSession(global_vector_db, global_teachability, model_client)
    agent_session.client_websocket = websocket
    await agent_session.initialize_agents()

    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"Received via WebSocket: {data}")
            await agent_session.handle_message(data, websocket)
    except WebSocketDisconnect: logger.info("WebSocket connection closed.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try: await websocket.send_json({"role": "error", "content": f"Server error: {str(e)}"})
        except: pass
    finally:
        if agent_session.input_future and not agent_session.input_future.done(): agent_session.input_future.cancel()
        logger.info("Cleaned up WebSocket session resources.")

@app.get("/")
async def get_root(): return {"message": "Personal AI Agent Backend running. WS at /ws/chat"}

if __name__ == "__main__": uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
