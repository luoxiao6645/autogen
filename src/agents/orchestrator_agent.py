from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelClient

class OrchestratorAgent(AssistantAgent):
    def __init__(self, name: str, model_client: ModelClient, system_message: str = None, **kwargs):
        DEFAULT_SYSTEM_MESSAGE = (
            "You are a helpful AI orchestrator. Your role is to understand user requests. "
            "If the user asks a question that requires looking up information from a knowledge base, "
            "use the 'answer_from_knowledge_base' tool available from the 'RetrievalExpert' agent to get the answer. "
            "Do not try to answer such questions yourself directly. "
            "For other types of requests, you can respond directly or plan further actions (though advanced planning is not yet implemented). "
            "When using tools, clearly indicate that you are using a tool and what information you are seeking. "
            "After receiving the information from the tool, present it clearly to the user."
        )
        super().__init__(
            name,
            model_client=model_client,
            system_message=system_message or DEFAULT_SYSTEM_MESSAGE,
            **kwargs
            # Note: For the orchestrator to *call* a tool from another agent,
            # that tool needs to be effectively in its "scope".
            # This is typically handled by:
            # 1. The other agent (RetrievalExpert) being part of a group chat where the orchestrator is the admin/speaker.
            # 2. The orchestrator initiating a chat with RetrievalExpert, and RetrievalExpert having the tool registered.
            # 3. Explicitly passing function definitions to the orchestrator's llm_config if it's making direct LLM calls
            #    that are expected to generate function calls for tools hosted elsewhere (more complex).
            # The `main.py` setup will handle how these agents interact and how tools are exposed.
        )

if __name__ == '__main__':
    import os
    from dotenv import load_dotenv
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    # from src.agents.retrieval_agent import InformationRetrievalAgent # For testing interaction
    # from src.agents.interaction_agent import UserInteractionAgent # For testing interaction

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("OPENAI_API_KEY not found. Please set it in your .env file or environment.")
    else:
        model_client_instance = OpenAIChatCompletionClient(model="gpt-4o", api_key=api_key) # Using a more capable model for orchestrator

        orchestrator = OrchestratorAgent(
            name="Orchestrator",
            model_client=model_client_instance
        )
        print(f"{orchestrator.name} initialized with system message: '{orchestrator.system_message}'")

        # --- Example of how Orchestrator might interact (conceptual) ---
        # This test requires RetrievalAgent to be set up with its tool and a UserProxyAgent to initiate.
        # This will be fully tested via the FastAPI backend.

        # 1. Setup RetrievalExpert with its tool (as in retrieval_agent.py)
        # from autogen_ext.memory.chromadb import ChromaDBVectorMemory, ChromaDBVectorMemoryConfig
        # chroma_config = ChromaDBVectorMemoryConfig(collection_name="test_orchestrator_interaction")
        # vector_db = ChromaDBVectorMemory(config=chroma_config)
        # # asyncio.run(vector_db.clear())
        # # asyncio.run(vector_db.add([MemoryContent(content="AutoGen is a Microsoft framework.", mime_type=MemoryMimeType.TEXT)]))

        # retrieval_expert = InformationRetrievalAgent(
        #     name="RetrievalExpert",
        #     model_client=OpenAIChatCompletionClient(model="gpt-3.5-turbo", api_key=api_key),
        #     vector_memory=vector_db
        # )

        # 2. Setup User Proxy
        # user_proxy_for_test = UserInteractionAgent(name="TestUser", human_input_mode="NEVER", max_consecutive_auto_reply=1)

        # 3. Orchestrator needs to know about RetrievalExpert's tool.
        # This can be done by adding RetrievalExpert to a group chat managed by Orchestrator,
        # or by directly registering RetrievalExpert's tool schema with the Orchestrator if they chat directly.

        # For a direct chat where Orchestrator calls RetrievalExpert's tool:
        # orchestrator.register_for_llm(name="answer_from_knowledge_base", description="Answers from KB.") # Not quite, this is for its own tools

        # The more common AutoGen pattern:
        # group_chat_manager = GroupChatManager(
        #     groupchat=GroupChat(agents=[user_proxy_for_test, orchestrator, retrieval_expert], messages=[]),
        #     llm_config={"model_client": model_client_instance} # Or orchestrator acts as llm_config provider
        # )
        # user_proxy_for_test.initiate_chat(group_chat_manager, message="What is AutoGen according to the knowledge base?")

        print("OrchestratorAgent initialized. Its ability to use RetrievalExpert's tool will be tested via the backend.")
