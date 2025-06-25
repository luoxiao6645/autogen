import asyncio
import logging
import os
from typing import Callable, Optional, Dict, List, Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from autogen_agentchat.agents import ConversableAgent, AssistantAgent, UserProxyAgent # Added AssistantAgent, UserProxyAgent
from autogen_core.models import ModelClient # Added ModelClient
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.memory.chromadb import ChromaDBVectorMemory, ChromaDBVectorMemoryConfig
from src.agents.interaction_agent import UserInteractionAgent
from src.agents.orchestrator_agent import OrchestratorAgent
from src.agents.retrieval_agent import InformationRetrievalAgent

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
        self.shared_vector_db = shared_vector_db
        self.model_client = model_client # Store the shared model client
        self.chat_initiated = False
        self.input_future: Optional[asyncio.Future] = None
        self.client_websocket: Optional[WebSocket] = None

    async def initialize_agents(self):
        if self.shared_vector_db:
            self.retrieval_agent = InformationRetrievalAgent(
                name="RetrievalExpert", # Name used in Orchestrator's prompt
                model_client=self.model_client, # Use the shared model client
                vector_memory=self.shared_vector_db
            )
            logger.info("InformationRetrievalAgent initialized.")
        else:
            logger.warning("Shared vector DB not available; InformationRetrievalAgent not fully initialized.")
            # This would be a critical failure for RAG

        # Orchestrator needs to be able to call the tool from RetrievalExpert
        # We register the tool method from retrieval_agent instance directly with orchestrator
        self.orchestrator = OrchestratorAgent(
            name="Orchestrator",
            model_client=self.model_client, # Use the shared model client
        )
        if self.retrieval_agent:
            # Register tools from retrieval_agent with the orchestrator
            # This allows orchestrator's LLM to generate tool calls for these tools.
            self.orchestrator.register_tools(
                tools=[self.retrieval_agent.answer_from_knowledge_base],
                # By default, execution happens on the agent where tool is registered (Orchestrator)
                # but the tool itself is a method of self.retrieval_agent, so it uses retrieval_agent's state.
            )
            logger.info("Registered RetrievalExpert's tools with Orchestrator.")


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
        logger.info("UserInteractionAgent and OrchestratorAgent initialized for session.")
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

            # Define a reply function that sends orchestrator's messages to the WebSocket
            async def send_orchestrator_reply_to_ws(
                messages: List[Dict[str, Any]], sender: ConversableAgent, recipient: ConversableAgent
            ) -> List[Dict[str, Any]]:
                last_message = messages[-1]
                logger.info(f"Orchestrator ({sender.name}) sending message to {recipient.name}: {last_message.get('content')}")

                # We only want to send to websocket if the recipient is the UserInteractionAgent (our proxy)
                if recipient.name == self.user_interaction_agent.name:
                    await websocket.send_json({
                        "role": "assistant",
                        "content": last_message.get("content", "")
                    })
                return messages # Important to return messages for AutoGen's internal processing

            # Register the reply function for the orchestrator
            # This will capture messages from the orchestrator when it's its turn to speak.
            self.orchestrator.register_reply_function(
                trigger=self.user_interaction_agent, # Trigger when orchestrator is replying to user_interaction_agent
                reply_function=send_orchestrator_reply_to_ws,
                # position=1 # Ensure it's called appropriately
            )

            # Also, UserInteractionAgent needs to know how to send its messages (the initial one)
            # to the orchestrator without printing to console.
            # The initiate_chat will handle the first message. Subsequent ones are via input_func.

            async def initiate_and_run_chat():
                # The tools registered on Orchestrator (which are methods of RetrievalExpert)
                # will be executed by the Orchestrator's tool_executor, which calls the methods.
                await self.user_interaction_agent.a_initiate_chat(
                    recipient=self.orchestrator,
                    clear_history=True,
                    message=message_content,
                    # Max turns for safety, can be adjusted
                    # max_turns=5
                )
                # If the chat ends and the last speaker was the orchestrator,
                # its final message should have been sent by send_orchestrator_reply_to_ws.
                # If the orchestrator is now waiting for input (e.g. after a tool call and summary),
                # the UserInteractionAgent's input_func (get_input_from_websocket) will be triggered.
                logger.info("Chat interaction with Orchestrator concluded or awaiting further input.")

            asyncio.create_task(initiate_and_run_chat())
        else:
            logger.warning(f"Received unexpected message: {message_content} while chat is already initiated and not waiting for specific input.")
            # This condition should ideally not be hit if input_future logic is correct.


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
        logger.error("Vector DB not available at session start.")
        await websocket.send_json({"role": "error", "content": "Knowledge base (Vector DB) is not available on the server."})
        await websocket.close(code=1008); return

    # Each connection gets its own model client and agent session for isolation
    model_client = OpenAIChatCompletionClient(model="gpt-4o", api_key=api_key) # Orchestrator uses gpt-4o

    agent_session = AgentSession(shared_vector_db=global_vector_db, model_client=model_client)
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) # Added reload=True for dev convenience
