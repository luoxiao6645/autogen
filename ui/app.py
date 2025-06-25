import streamlit as st
import websockets
import asyncio
import json
import os

# --- Configuration ---
WEBSOCKET_URI = os.getenv("WEBSOCKET_URI", "ws://localhost:8000/ws/chat")

st.title("My Personal AI Agent")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [] # For displaying chat history
if "websocket_connected" not in st.session_state:
    st.session_state.websocket_connected = False
if "websocket_connection" not in st.session_state:
    st.session_state.websocket_connection = None
if "agent_is_waiting_for_input" not in st.session_state:
    st.session_state.agent_is_waiting_for_input = False
if "last_input_prompt" not in st.session_state:
    st.session_state.last_input_prompt = "What can I help you with?"


# --- WebSocket Connection Management ---
async def ensure_websocket_connection():
    if not st.session_state.websocket_connected or st.session_state.websocket_connection is None:
        try:
            st.session_state.websocket_connection = await websockets.connect(WEBSOCKET_URI)
            st.session_state.websocket_connected = True
            st.info("Connected to agent backend.")
            # Start a task to listen for messages from the backend
            asyncio.create_task(receive_messages(st.session_state.websocket_connection))
        except Exception as e:
            st.error(f"Failed to connect to WebSocket: {e}")
            st.session_state.websocket_connected = False
            st.session_state.websocket_connection = None

async def close_websocket_connection():
    if st.session_state.websocket_connection:
        await st.session_state.websocket_connection.close()
    st.session_state.websocket_connected = False
    st.session_state.websocket_connection = None
    st.info("Disconnected from agent backend.")

# --- Message Handling ---
async def send_message_to_backend(message_content: str):
    if st.session_state.websocket_connected and st.session_state.websocket_connection:
        try:
            await st.session_state.websocket_connection.send(message_content)
        except Exception as e:
            st.error(f"Error sending message: {e}")
            # Attempt to reconnect or handle error
            await close_websocket_connection() # simple close on error
            st.rerun()
    else:
        st.warning("Not connected to agent backend. Attempting to reconnect.")
        await ensure_websocket_connection()
        # Optionally, resend the message here if connection is successful,
        # or ask user to try again. For now, user has to resubmit.

async def receive_messages(websocket: websockets.WebSocketClientProtocol):
    try:
        async for message_str in websocket:
            try:
                message_data = json.loads(message_str)
                role = message_data.get("role", "system")
                content = message_data.get("content", "")
                msg_type = message_data.get("type")

                st.session_state.messages.append({"role": role, "content": content, "type": msg_type})

                if msg_type == "input_request":
                    st.session_state.agent_is_waiting_for_input = True
                    st.session_state.last_input_prompt = content # Update placeholder for next input
                else:
                    st.session_state.agent_is_waiting_for_input = False

                st.rerun() # Rerun Streamlit to update the UI with the new message
            except json.JSONDecodeError:
                st.warning(f"Received non-JSON message: {message_str}")
            except Exception as e:
                st.error(f"Error processing received message: {e}")
                # Decide if we should break or try to continue
    except websockets.exceptions.ConnectionClosed:
        st.warning("Connection to agent backend closed.")
        st.session_state.websocket_connected = False
        st.session_state.websocket_connection = None
        st.rerun()
    except Exception as e:
        st.error(f"WebSocket error: {e}")
        st.session_state.websocket_connected = False
        st.session_state.websocket_connection = None
        st.rerun()

# --- UI Rendering ---

# Attempt to connect on first app load
if not st.session_state.websocket_connected:
    # Using st.button to trigger async connection in Streamlit's execution model
    if st.button("Connect to Agent Backend"):
        asyncio.run(ensure_websocket_connection())
        st.rerun() # Rerun to update UI based on connection status
else:
    if st.button("Disconnect"):
        asyncio.run(close_websocket_connection())
        st.rerun()


# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("type") == "input_request" and msg["role"] == "system":
            st.caption("Agent is waiting for your input above.")


# Chat input field
chat_input_placeholder = st.session_state.last_input_prompt if st.session_state.agent_is_waiting_for_input else "What can I help you with?"
user_prompt = st.chat_input(chat_input_placeholder, disabled=not st.session_state.websocket_connected)

if user_prompt:
    if st.session_state.websocket_connected:
        # Display user message immediately
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Send message to backend
        asyncio.run(send_message_to_backend(user_prompt))
        st.session_state.agent_is_waiting_for_input = False # Assume agent will reply or request new input
        st.session_state.last_input_prompt = "What can I help you with?" # Reset placeholder
        # No st.rerun() here, receive_messages will trigger it
    else:
        st.warning("Please connect to the agent backend first.")

# --- Sidebar ---
st.sidebar.info(
    "Personal AI Agent Interface. \n"
    "Connect to the backend, then start chatting. \n"
    f"Backend URI: {WEBSOCKET_URI}"
)
if st.session_state.websocket_connected:
    st.sidebar.success("Connected to WebSocket.")
else:
    st.sidebar.error("Disconnected from WebSocket.")

if st.session_state.agent_is_waiting_for_input:
    st.sidebar.warning("Agent is waiting for your input.")
