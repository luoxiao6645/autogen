from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelClient

class OrchestratorAgent(AssistantAgent):
    def __init__(self, name: str, model_client: ModelClient, system_message: str = None, **kwargs):
        DEFAULT_SYSTEM_MESSAGE = (
            "You are a helpful AI orchestrator. Your role is to understand user requests. "
            "If the user asks a question that requires looking up information from a knowledge base, "
            "use the 'answer_from_knowledge_base' tool available from the 'RetrievalExpert' agent to get the answer. "
            "If the user asks you to perform an action, like sending an email, "
            "use the 'send_email' tool available from the 'TaskExecutor' agent. " # Changed from send_mock_email
            "Provide all necessary arguments for the tools, such as recipient, subject, and body for emails. "
            "Do not try to answer information queries yourself directly if they should come from the knowledge base. "
            "Do not try to perform actions yourself directly if a tool is available. "
            "When using tools, clearly indicate what you are doing. "
            "After receiving information or confirmation from a tool, present it clearly to the user."
        )
        super().__init__(
            name,
            model_client=model_client,
            system_message=system_message or DEFAULT_SYSTEM_MESSAGE,
            **kwargs
        )

if __name__ == '__main__':
    import os
    from dotenv import load_dotenv
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("OPENAI_API_KEY not found. Please set it in your .env file or environment.")
    else:
        model_client_instance = OpenAIChatCompletionClient(model="gpt-4o", api_key=api_key)

        orchestrator = OrchestratorAgent(
            name="Orchestrator",
            model_client=model_client_instance
        )
        print(f"{orchestrator.name} initialized with system message: '{orchestrator.system_message}'")
        print("OrchestratorAgent initialized. Its ability to use tools will be tested via the backend.")
