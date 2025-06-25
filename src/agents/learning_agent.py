from typing import Optional, List, Dict, Any, Annotated

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelClient
from autogen_core.tools import tool # Added for explicit recall tool
# Teachability related imports
from autogen_ext.models.openai import OpenAIChatCompletionClient as DefaultClient
from autogen_ext.experimental.task_centric_memory import MemoryController, Teachability

class LearningMemoryAgent(AssistantAgent):
    def __init__(
        self,
        name: str,
        model_client: ModelClient,
        teachability: Teachability,
        system_message: Optional[str] = None,
        **kwargs
    ):
        DEFAULT_SYSTEM_MESSAGE = (
            "You are an AI assistant with the ability to learn from conversations and recall learned information. "
            "When the user tells you something to remember (e.g., 'My name is...', 'My favorite color is...', 'Remember this fact: ...'), "
            "your underlying Teachability mechanism will automatically try to store this information. You should acknowledge what you have learned. "
            "When asked a question, or when information you previously learned seems relevant, use the 'recall_learned_information' tool to fetch it. "
            "Clearly state when you are using recalled information."
        )

        super().__init__(
            name,
            model_client=model_client,
            system_message=system_message or DEFAULT_SYSTEM_MESSAGE,
            memory=teachability,
            **kwargs
        )
        self._teachability = teachability # Keep a reference if direct calls to memory are needed
        self.register_tools([self.recall_learned_information])

    @tool()
    async def recall_learned_information(self, topic_or_question: Annotated[str, "The specific topic or question to recall information about."]) -> str:
        """
        Recalls information that was previously learned and stored in memory about a specific topic or question.
        Use this tool to answer questions based on prior teachings or to retrieve user preferences.
        """
        print(f"[LearningMemoryAgent - Recall Tool Called] Recalling information about: {topic_or_question}")
        if not self._teachability or not self._teachability.memory_controller:
            return "Learning memory is not properly configured."

        # Teachability doesn't have a direct 'query' method.
        # MemoryController has 'get_memory_as_str' which retrieves recent raw memories,
        # or we can simulate a query by sending a message to ourselves (or a helper agent)
        # that would trigger Teachability's recall if the LLM deems it relevant.

        # For a more direct recall from MemoryController (if it has a suitable query method):
        # This part is a bit conceptual as MemoryController's direct query API isn't as rich as a vector DB.
        # It primarily works by providing context to an LLM.
        # Let's try to get recent memories and let this agent's LLM filter/use them.

        # memory_str = self._teachability.memory_controller.get_memory_as_str()
        # if not memory_str:
        #     return f"I don't have any learned information about '{topic_or_question}' yet."

        # Forcing a recall through an LLM call with the context of learned memories
        # This is how Teachability usually makes learned info available.
        # The agent's own system message and the `memory=teachability` already set this up.
        # So, the tool can just prompt itself with the topic.

        # We will ask this agent (itself) to answer the question, relying on Teachability
        # to inject relevant learned context.
        response_message = await self.generate_chat_reply(
            messages=[{"role": "user", "content": f"Based on what you've learned previously, what do you know about: {topic_or_question}?"}],
        )

        recalled_info = "Could not recall specific information on that topic."
        if isinstance(response_message, str):
            recalled_info = response_message
        elif isinstance(response_message, dict) and "content" in response_message:
            recalled_info = response_message["content"]

        # Avoid self-referential statements like "Based on what I've learned..." in the final tool output
        # if it's too verbose. The LLM calling this tool will handle phrasing.
        # We might need to refine the prompt to `generate_chat_reply` or post-process `recalled_info`.
        # For now, returning the direct answer.
        if "don't have any learned information" in recalled_info or "don't know anything specific" in recalled_info:
             print(f"[LearningMemoryAgent - Recall Tool] No specific info found for: {topic_or_question}")
        else:
             print(f"[LearningMemoryAgent - Recall Tool] Recalled for '{topic_or_question}': {recalled_info}")
        return recalled_info


if __name__ == '__main__':
    import os
    import asyncio
    from dotenv import load_dotenv
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    from autogen_agentchat.agents import UserProxyAgent

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    async def test_learning_agent_with_recall_tool():
        if not api_key:
            print("OPENAI_API_KEY not found.")
            return

        learning_agent_llm = OpenAIChatCompletionClient(model="gpt-4o", api_key=api_key) # Use GPT-4o for better recall/tool use
        memory_controller_llm = OpenAIChatCompletionClient(model="gpt-3.5-turbo", api_key=api_key)

        memory_controller = MemoryController(reset=True, client=memory_controller_llm)
        teachability = Teachability(memory_controller=memory_controller, reset_memory=True)

        learning_agent = LearningMemoryAgent(
            name="LearnerWithRecall", model_client=learning_agent_llm, teachability=teachability)
        print(f"{learning_agent.name} initialized.")

        user_proxy = UserProxyAgent(name="TestUserForRecall", human_input_mode="NEVER", max_consecutive_auto_reply=0)

        # --- Teach something ---
        await user_proxy.a_initiate_chat(recipient=learning_agent, message="Please remember my user ID is user123.", max_turns=2)
        await user_proxy.a_initiate_chat(recipient=learning_agent, message="And my preferred contact method is email.", max_turns=2, clear_history=False)

        print("\n--- Testing Recall Tool directly on Learner agent ---")
        # This tests if the tool itself works. Orchestrator would call this via LLM.
        recalled_id = await learning_agent.recall_learned_information("user ID")
        print(f"Recalled User ID (direct tool call): {recalled_id}")

        recalled_contact = await learning_agent.recall_learned_information("preferred contact method")
        print(f"Recalled Contact Method (direct tool call): {recalled_contact}")

        recalled_nonexistent = await learning_agent.recall_learned_information("favorite food")
        print(f"Recalled Non-existent Info (direct tool call): {recalled_nonexistent}")

        # --- Test Orchestrator (simulated) telling Learner to use its tool ---
        # This requires Orchestrator to be configured to call Learner's tool.
        # For this unit test, we'll have a UserProxyAgent act as Orchestrator

        orchestrator_sim_llm = OpenAIChatCompletionClient(model="gpt-4o", api_key=api_key)
        orchestrator_sim = UserProxyAgent(
            name="OrchestratorSim",
            human_input_mode="NEVER",
            model_client=orchestrator_sim_llm,
            system_message="You need to find out the user's ID. Use the 'recall_learned_information' tool from the 'LearnerWithRecall' agent."
        )
        # Register Learner's tool with the simulated Orchestrator
        orchestrator_sim.register_tools(tools=[learning_agent.recall_learned_information])

        print("\n--- Simulating Orchestrator calling Learner's recall tool ---")
        await orchestrator_sim.a_initiate_chat(
            recipient=learning_agent, # The chat is with Learner, but Orchestrator's LLM calls the tool
            message="What is the user's ID that was previously learned?",
            max_turns=3 # Orchestrator asks, LLM generates tool call, tool executes, LLM summarizes
        )
        chat_history = orchestrator_sim.chat_messages_for_summary(learning_agent)
        if chat_history:
             print(f"Orchestrator-Learner recall conversation history:")
             for msg in chat_history:
                 print(f"  Role: {msg.get('role')}, Name: {msg.get('name')}, Content: {msg.get('content')}")
                 if msg.get("tool_calls"): print(f"    Tool Calls: {msg.get('tool_calls')}")
                 if msg.get("tool_responses"): print(f"    Tool Responses: {msg.get('tool_responses')}")


    if __name__ == '__main__':
        asyncio.run(test_learning_agent_with_recall_tool())
