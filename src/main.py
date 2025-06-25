import asyncio
import logging
import os
from typing import Callable, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from autogen_agentchat.agents import ConversableAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from src.agents.interaction_agent import UserInteractionAgent
from src.agents.orchestrator_agent import OrchestratorAgent

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global store for agents, this is simplistic for a single-user demo.
# For multi-user, you'd need session management.
# Also, agent instances should ideally be managed per connection/session.
class AgentSession:
    def __init__(self):
        self.user_interaction_agent: Optional[UserInteractionAgent] = None
        self.orchestrator: Optional[OrchestratorAgent] = None
        self.chat_initiated = False
        self.input_future: Optional[asyncio.Future] = None
        self.client_websocket: Optional[WebSocket] = None

    async def initialize_agents(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY not found.")
            if self.client_websocket:
                await self.client_websocket.send_json({
                    "role": "error",
                    "content": "OpenAI API key not configured on the server."
                })
            return False

        model_client = OpenAIChatCompletionClient(model="gpt-3.5-turbo", api_key=api_key)

        self.orchestrator = OrchestratorAgent(
            name="Orchestrator",
            model_client=model_client
        )

        # This function will be called by UserInteractionAgent when it needs input
        async def get_input_from_websocket(prompt: str) -> str:
            logger.info(f"Agent is requesting input: {prompt}")
            if self.client_websocket:
                # Send a special message to UI indicating agent is waiting for input
                await self.client_websocket.send_json({
                    "role": "system", # Or a custom role like "input_request"
                    "content": prompt, # The prompt from the agent
                    "type": "input_request"
                })

            self.input_future = asyncio.get_event_loop().create_future()
            try:
                # Wait for the input to be set by the websocket receiver
                user_response = await asyncio.wait_for(self.input_future, timeout=300) # 5 min timeout
                return user_response
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for user input via WebSocket.")
                return "No input received due to timeout."


        self.user_interaction_agent = UserInteractionAgent(
            name="UserWebAppProxy",
            human_input_mode="ALWAYS", # Will always call our input_func
            input_func=get_input_from_websocket,
            # max_consecutive_auto_reply is not as relevant when human_input_mode is ALWAYS for the primary user proxy
        )
        logger.info("Agents initialized for session.")
        return True

    async def handle_message(self, message_content: str, websocket: WebSocket):
        if not self.user_interaction_agent or not self.orchestrator:
            logger.error("Agents not initialized.")
            await websocket.send_json({"role": "error", "content": "Agents not initialized on server."})
            return

        if self.input_future and not self.input_future.done():
            # If agent is waiting for input, this new message is the input
            self.input_future.set_result(message_content)
            # The agent's conversation flow will resume via get_input_from_websocket
        elif not self.chat_initiated:
            self.chat_initiated = True
            logger.info(f"Initiating chat with orchestrator. Initial message: {message_content}")

            # Define a custom send function for the agents to use the websocket
            def ws_send(sender: ConversableAgent, message_dict: dict, recipient: ConversableAgent):
                # By default, UserProxyAgent (and AssistantAgent) print to console or use registered reply funcs.
                # We want to intercept messages *from* the orchestrator *to* the user_interaction_agent (proxy)
                # and send them to the WebSocket client.
                # The user_interaction_agent itself won't "speak" unless its input_func is called or it's summarizing.

                # We are interested in messages from the orchestrator
                if sender.name == self.orchestrator.name:
                    logger.info(f"Orchestrator sending message to {recipient.name}: {message_dict.get('content')}")
                    asyncio.create_task(
                        websocket.send_json({
                            "role": "assistant", # Orchestrator is the assistant in this context
                            "content": message_dict.get("content", "")
                        })
                    )

            # Register the custom send function for the orchestrator
            self.orchestrator.register_reply([self.user_interaction_agent, ConversableAgent.Sender], ws_send)
            # Also for user_interaction_agent if it needs to send non-input-request messages
            # self.user_interaction_agent.register_reply(ConversableAgent.Sender, ws_send_from_user_proxy)


            # Run the chat initiation in a background task so it doesn't block the websocket
            async def initiate_and_run_chat():
                await self.user_interaction_agent.a_initiate_chat(
                    recipient=self.orchestrator,
                    clear_history=True,
                    message=message_content,
                )
                # After chat concludes or if more input is needed, input_func will be called.
                # If chat ends naturally by orchestrator, it might send a final message via ws_send.
                # If orchestrator needs more input, get_input_from_websocket will be triggered.
                logger.info("Chat interaction (or first part of it) concluded.")

            asyncio.create_task(initiate_and_run_chat())
        else:
            # This case should ideally be handled by input_func if chat is ongoing
            logger.warning(f"Received unexpected message: {message_content} while chat is already initiated and not waiting for specific input.")
            await websocket.send_json({"role": "system", "content": "Received your message, but unsure how to proceed in current state."})


@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established.")

    agent_session = AgentSession()
    agent_session.client_websocket = websocket # Store websocket for the session

    if not await agent_session.initialize_agents():
        await websocket.close(code=1008) # Policy Violation or similar
        return

    try:
        while True:
            data = await websocket.receive_text()
            # Assuming incoming data is just the message string for simplicity for now.
            # In a real app, this might be JSON with more structure.
            logger.info(f"Received message via WebSocket: {data}")
            await agent_session.handle_message(data, websocket)

    except WebSocketDisconnect:
        logger.info("WebSocket connection closed.")
    except Exception as e:
        logger.error(f"Error in WebSocket endpoint: {e}", exc_info=True)
        try:
            await websocket.send_json({"role": "error", "content": f"Server error: {str(e)}"})
        except: # If sending fails, socket might already be closed
            pass
    finally:
        if agent_session.input_future and not agent_session.input_future.done():
            agent_session.input_future.cancel() # Clean up future if client disconnects while agent is waiting
        logger.info("Cleaned up agent session resources.")


@app.get("/")
async def get_root():
    return {"message": "Personal AI Agent Backend is running. Connect via WebSocket at /ws/chat"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
