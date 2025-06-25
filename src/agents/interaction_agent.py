from autogen_agentchat.agents import UserProxyAgent

class UserInteractionAgent(UserProxyAgent):
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)

if __name__ == '__main__':
    # Example usage (optional, for direct testing of this agent)
    ui_agent = UserInteractionAgent(name="UserAgent")

    # To test, you would typically initiate a chat from this agent to another,
    # or have another agent initiate a chat with it.
    # For example:
    # assistant = AssistantAgent("assistant", llm_config={"model": "gpt-4"})
    # ui_agent.initiate_chat(assistant, message="Hello, Assistant!")
    print(f"{ui_agent.name} initialized. This agent is typically used to get user input or act as a proxy for the user in a chat.")
