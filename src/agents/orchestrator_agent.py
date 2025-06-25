from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelClient

class OrchestratorAgent(AssistantAgent):
    def __init__(self, name: str, model_client: ModelClient, system_message: str = None, **kwargs):
        DEFAULT_SYSTEM_MESSAGE = (
            "You are a helpful AI orchestrator. Your role is to understand user requests and coordinate actions. "
            "1. **Information Retrieval**: If the user asks a question that requires looking up factual information "
            "   from a knowledge base, use the 'answer_from_knowledge_base' tool available from the 'RetrievalExpert' agent. "
            "2. **Task Execution**: If the user asks you to perform an action, like sending an email, "
            "   use the 'send_email' tool available from the 'TaskExecutor' agent. "
            "3. **Learning & Personalization**: "
            "   - **Teaching**: If the user explicitly tells you to remember something (e.g., 'My name is...', 'Remember this fact...', 'My preference is...'), "
            "     acknowledge this and inform the 'Learner' agent by initiating a brief chat with it, passing on the information clearly. For example, send a message to Learner: 'The user wants us to remember: [information to remember]'. "
            "   - **Recall**: If a user's query or your task could benefit from previously learned personal information or preferences, "
            "     use the 'recall_learned_information' tool available from the 'Learner' agent to fetch this information. For example, ask: 'What is the user's name?' or 'What is the user's preferred [topic]?'. "
            "Provide all necessary arguments for tools. Do not try to answer information queries or perform actions yourself if a specific tool or expert agent is available. "
            "When using tools or interacting with other agents, clearly indicate what you are doing. "
            "After receiving information or confirmation, synthesize it and present it clearly to the user."
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
        print("OrchestratorAgent initialized. Its ability to use tools and interact with other agents will be tested via the backend.")
