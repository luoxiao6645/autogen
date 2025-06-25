from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelClient

class OrchestratorAgent(AssistantAgent):
    def __init__(self, name: str, model_client: ModelClient, system_message: str = None, **kwargs):
        DEFAULT_SYSTEM_MESSAGE = (
            "You are a helpful AI orchestrator. Your role is to understand user requests, "
            "coordinate with other specialized agents if necessary (though not implemented yet), "
            "and provide a coherent final response. For now, acknowledge the user's request."
        )
        super().__init__(
            name,
            model_client=model_client,
            system_message=system_message or DEFAULT_SYSTEM_MESSAGE,
            **kwargs
        )

if __name__ == '__main__':
    # This section is for optional, direct testing of the OrchestratorAgent.
    # It requires an OpenAI API key to be set in the environment or passed via llm_config.
    import os
    from dotenv import load_dotenv
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("OPENAI_API_KEY not found. Please set it in your .env file or environment.")
    else:
        # Example of creating a model client (replace with your actual configuration)
        model_client_instance = OpenAIChatCompletionClient(model="gpt-3.5-turbo", api_key=api_key)

        orchestrator = OrchestratorAgent(
            name="Orchestrator",
            model_client=model_client_instance
        )
        print(f"{orchestrator.name} initialized with system message: '{orchestrator.system_message}'")

        # To test interaction (requires another agent, like a UserProxyAgent):
        # from src.agents.interaction_agent import UserInteractionAgent # Assuming it's in the path
        # user_proxy = UserInteractionAgent(name="TestUserProxy", human_input_mode="NEVER", max_consecutive_auto_reply=1)
        # user_proxy.initiate_chat(orchestrator, message="Hello Orchestrator, what can you do?")

        # For now, just confirming initialization.
        # Actual interaction will be tested in the end-to-end step.
