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
from src.agents.interaction_agent import UserInteractionAgent
from src.agents.orchestrator_agent import OrchestratorAgent
from src.agents.retrieval_agent import InformationRetrievalAgent
from src.agents.task_agent import TaskExecutionAgent

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PERSISTENCE_DIR = "./chroma_db_store"
COLLECTION_NAME = "personal_agent_docs"
global_vector_db: Optional[ChromaDBVectorMemory] = None

@app.on_event("startup")
async def startup_event():
    global global_vector_db
    logger.info(f"FastAPI startup: Initializing ChromaDBVectorMemory for collection '{COLLECTION_NAME}' from '{PERSISTENCE_DIR}'...")
    if not os.path.exists(PERSISTENCE_DIR):
        logger.warning(f"ChromaDB persistence directory '{PERSISTENCE_DIR}' not found. Indexing may not have been run.")

    chroma_config = ChromaDBVectorMemoryConfig(
        collection_name=COLLECTION_NAME,
        persistence_path=PERSISTENCE_DIR,
    )
    try:
        global_vector_db = ChromaDBVectorMemory(config=chroma_config)
        logger.info(f"ChromaDBVectorMemory for collection '{COLLECTION_NAME}' initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDBVectorMemory: {e}", exc_info=True)
        global_vector_db = None

class AgentSession:
    def __init__(self, shared_vector_db: ChromaDBVectorMemory, model_client: ModelClient):
        self.user_interaction_agent: Optional[UserInteractionAgent] = None
        self.orchestrator: Optional[OrchestratorAgent] = None
        self.retrieval_agent: Optional[InformationRetrievalAgent] = None
        self.task_agent: Optional[TaskExecutionAgent] = None
        self.shared_vector_db = shared_vector_db
        self.model_client = model_client
        self.chat_initiated = False
        self.input_future: Optional[asyncio.Future] = None
        self.client_websocket: Optional[WebSocket] = None

    async def initialize_agents(self):
        # Initialize Retrieval Agent
        if self.shared_vector_db:
            self.retrieval_agent = InformationRetrievalAgent(
                name="RetrievalExpert",
                model_client=self.model_client,
                vector_memory=self.shared_vector_db
            )
            logger.info("InformationRetrievalAgent initialized.")
        else:
            logger.warning("Shared vector DB not available; InformationRetrievalAgent not fully initialized.")

        # Initialize Task Execution Agent
        self.task_agent = TaskExecutionAgent(
            name="TaskExecutor",
            model_client=self.model_client
        )
        logger.info("TaskExecutionAgent initialized.")

        # Initialize Orchestrator Agent
        self.orchestrator = OrchestratorAgent(
            name="Orchestrator",
            model_client=self.model_client,
        )

        tools_to_register = []
        if self.retrieval_agent:
            tools_to_register.append(self.retrieval_agent.answer_from_knowledge_base)
            logger.info("Prepared RetrievalExpert's tool for Orchestrator.")

        if self.task_agent:
            tools_to_register.append(self.task_agent.send_email) # Changed from send_mock_email
            logger.info("Prepared TaskExecutor's 'send_email' tool for Orchestrator.")

        if tools_to_register:
            self.orchestrator.register_tools(tools=tools_to_register)
            logger.info(f"Registered {len(tools_to_register)} tools with Orchestrator.")


        async def get_input_from_websocket(prompt: str) -> str:
            logger.info(f"Agent is requesting input: {prompt}")
            if self.client_websocket:
                await self.client_websocket.send_json({
                    "role": "system", "content": prompt, "type": "input_request"
                })
            self.input_future = asyncio.get_event_loop().create_future()
            try:
                user_response = await asyncio.wait_for(self.input_future, timeout=300)
                return user_response
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for user input via WebSocket.")
                return "No input received due to timeout."

        self.user_interaction_agent = UserInteractionAgent(
            name="UserWebAppProxy",
            human_input_mode="ALWAYS",
            input_func=get_input_from_websocket,
        )
        logger.info("UserInteractionAgent and other core agents initialized for session.")
        return True

    async def handle_message(self, message_content: str, websocket: WebSocket):
        if not self.user_interaction_agent or not self.orchestrator:
            logger.error("Core agents not initialized.")
            await websocket.send_json({"role": "error", "content": "Core agents not initialized on server."})
            return

        if self.input_future and not self.input_future.done():
            self.input_future.set_result(message_content)
        elif not self.chat_initiated:
            self.chat_initiated = True
            logger.info(f"Initiating chat with orchestrator. Initial message: {message_content}")

            async def send_orchestrator_reply_to_ws(
                messages: List[Dict[str, Any]], sender: ConversableAgent, recipient: ConversableAgent
            ) -> List[Dict[str, Any]]:
                last_message = messages[-1]
                if recipient.name == self.user_interaction_agent.name and sender.name == self.orchestrator.name :
                    logger.info(f"Orchestrator ({sender.name}) sending message to UI via {recipient.name}: {last_message.get('content')}")
                    await websocket.send_json({
                        "role": "assistant",
                        "content": last_message.get("content", "")
                    })
                return messages

            self.orchestrator.register_reply_function(
                trigger=self.user_interaction_agent,
                reply_function=send_orchestrator_reply_to_ws,
            )

            async def initiate_and_run_chat():
                await self.user_interaction_agent.a_initiate_chat(
                    recipient=self.orchestrator,
                    clear_history=True,
                    message=message_content,
                )
                logger.info("Chat interaction with Orchestrator concluded or awaiting further input.")

            asyncio.create_task(initiate_and_run_chat())
        else:
            logger.warning(f"Received unexpected message: {message_content} while chat is already initiated and not waiting for specific input.")


@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found at session start.")
        await websocket.send_json({"role": "error", "content": "OpenAI API key not configured on the server."})
        await websocket.close(code=1008); return

    if global_vector_db is None:
        logger.warning("Vector DB not available at session start. RAG queries may fail.")

    model_client_gpt4o = OpenAIChatCompletionClient(model="gpt-4o", api_key=api_key)

    agent_session = AgentSession(shared_vector_db=global_vector_db, model_client=model_client_gpt4o)
    agent_session.client_websocket = websocket
    await agent_session.initialize_agents()

    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"Received message via WebSocket: {data}")
            await agent_session.handle_message(data, websocket)
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed.")
    except Exception as e:
        logger.error(f"Error in WebSocket endpoint: {e}", exc_info=True)
        try: await websocket.send_json({"role": "error", "content": f"Server error: {str(e)}"})
        except: pass
    finally:
        if agent_session.input_future and not agent_session.input_future.done():
            agent_session.input_future.cancel()
        logger.info("Cleaned up agent session resources for this WebSocket connection.")

@app.get("/")
async def get_root():
    return {"message": "Personal AI Agent Backend is running. Connect via WebSocket at /ws/chat"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
