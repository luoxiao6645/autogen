import asyncio # Added for async tool definition
from typing import Optional, List, Dict, Any
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelClient
from autogen_ext.memory.chromadb import ChromaDBVectorMemory
from autogen_core.memory import MemoryContent, MemoryMimeType # For testing
from autogen_core.tools import tool # For defining tools

class InformationRetrievalAgent(AssistantAgent):
    def __init__(
        self,
        name: str,
        model_client: ModelClient,
        vector_memory: ChromaDBVectorMemory,
        system_message: Optional[str] = None,
        n_results_for_rag: int = 3, # How many chunks to retrieve
        **kwargs
    ):
        DEFAULT_SYSTEM_MESSAGE = (
            "You are a specialized AI assistant for information retrieval. "
            "When you use the 'answer_from_knowledge_base' tool, you will be given a user query. "
            "Your tool will retrieve relevant context from the knowledge base. "
            "You must then synthesize an answer based *only* on this retrieved context and the user's query. "
            "If the context does not contain the answer, state that the information is not available in the current knowledge base. "
            "Do not use any external knowledge or make assumptions beyond the provided context. "
            "Be concise and directly answer the query using the information from the context."
        )

        super().__init__(
            name,
            model_client=model_client,
            system_message=system_message or DEFAULT_SYSTEM_MESSAGE,
            **kwargs
        )
        self._vector_memory = vector_memory
        self._n_results_for_rag = n_results_for_rag

        # Register the tool method with the agent.
        # The tool decorator handles schema generation.
        self.register_tools([self.answer_from_knowledge_base])

    @tool()
    async def answer_from_knowledge_base(self, user_query: str) -> str:
        """
        Answers a user's query based on information retrieved from the knowledge base.
        Use this tool when you need to find factual information to respond to a user.
        Args:
            user_query (str): The user's original query.
        Returns:
            str: The answer synthesized from the knowledge base context, or a statement that the information was not found.
        """
        if not self._vector_memory:
            return "Knowledge base (vector memory) is not configured for this agent."

        print(f"RetrievalExpert Tool: Received query '{user_query}' for knowledge base.")

        # 1. Query the vector memory
        try:
            # Assuming vector_memory.query returns a list of MemoryContent or similar objects
            # The query method in ChromaDBVectorMemory takes query_texts (list) and returns a list of lists of results.
            # For a single query, we expect a list containing one list of results.
            query_results_list = await self._vector_memory.query(query_texts=[user_query], n_results=self._n_results_for_rag)

            retrieved_chunks_content: List[str] = []
            if query_results_list and query_results_list[0]: # Check if the first query had results
                for item in query_results_list[0]: # Iterate through results for the first query
                    if item and item.content:
                         # Ensure content is not None and strip any potential surrounding quotes if they are strings
                        content_str = item.content
                        if isinstance(content_str, str):
                            retrieved_chunks_content.append(content_str.strip("'\""))
                        elif isinstance(content_str, list): # if content itself is a list of strings
                             retrieved_chunks_content.extend([str(c).strip("'\"") for c in content_str])


            if not retrieved_chunks_content:
                print("RetrievalExpert Tool: No relevant context found in the knowledge base.")
                return "I could not find relevant information in the current knowledge base to answer your query."

            retrieved_context = "\n\n".join(retrieved_chunks_content)
            print(f"RetrievalExpert Tool: Retrieved context:\n{retrieved_context[:500]}...") # Print first 500 chars

        except Exception as e:
            print(f"RetrievalExpert Tool: Error querying knowledge base: {e}")
            return "There was an error accessing the knowledge base."

        # 2. Synthesize an answer using the agent's LLM, based *only* on the retrieved context.
        # The system message of this agent already guides it to do this.
        # We need to make a new call to this agent's LLM with the context and query.

        # Construct a specific prompt for this internal LLM call
        synthesis_prompt = (
            f"Please answer the following user query based *only* on the provided context.\n\n"
            f"User Query: {user_query}\n\n"
            f"Retrieved Context:\n{retrieved_context}\n\n"
            "Answer:"
        )

        print(f"RetrievalExpert Tool: Sending to LLM for synthesis with prompt:\n{synthesis_prompt[:500]}...")

        try:
            # Create a temporary list of messages for this specific synthesis task
            # The system message of the agent is automatically included by run() or create_chat_completion_client().
            # We just need to provide the user-role message containing our constructed prompt.
            response_message = await self.generate_chat_reply(
                messages=[{"role": "user", "content": synthesis_prompt}],
                # No sender needed here as it's an internal call for the tool
            )

            # The response_message from generate_chat_reply could be a string or a dict.
            # If it's a dict, the content is usually in response_message['content']
            final_answer = ""
            if isinstance(response_message, str):
                final_answer = response_message
            elif isinstance(response_message, dict) and "content" in response_message:
                final_answer = response_message["content"]
            else:
                final_answer = "Could not synthesize an answer from the context."

            print(f"RetrievalExpert Tool: Synthesized answer: {final_answer}")
            return final_answer
        except Exception as e:
            print(f"RetrievalExpert Tool: Error during LLM synthesis: {e}")
            return "There was an error synthesizing the answer from the knowledge base."


if __name__ == '__main__':
    import os
    import asyncio
    from dotenv import load_dotenv
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    from autogen_ext.memory.chromadb import ChromaDBVectorMemory, ChromaDBVectorMemoryConfig
    from autogen_core.memory import MemoryContent, MemoryMimeType

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    async def test_retrieval_agent_tool():
        if not api_key:
            print("OPENAI_API_KEY not found. Please set it in your .env file or environment.")
            return

        model_client_instance = OpenAIChatCompletionClient(model="gpt-3.5-turbo", api_key=api_key)

        chroma_config = ChromaDBVectorMemoryConfig(
            collection_name="test_retrieval_tool_collection",
        )
        vector_db = ChromaDBVectorMemory(config=chroma_config)

        await vector_db.clear()
        await vector_db.add(
            [MemoryContent(content="AutoGen is a framework by Microsoft for building multi-agent applications.", mime_type=MemoryMimeType.TEXT, metadata={"source": "doc1"})]
        )
        print("Test vector DB populated for tool test.")

        retrieval_agent_for_tool_test = InformationRetrievalAgent(
            name="RetrievalExpertWithTool",
            model_client=model_client_instance,
            vector_memory=vector_db
        )
        print(f"{retrieval_agent_for_tool_test.name} initialized.")

        # Test the tool directly
        test_query = "What is AutoGen?"
        print(f"\nTesting tool with query: '{test_query}'")
        answer = await retrieval_agent_for_tool_test.answer_from_knowledge_base(user_query=test_query)
        print(f"\nAnswer from tool: {answer}")

        test_query_no_context = "What is the color of the sky?"
        print(f"\nTesting tool with query (no context expected): '{test_query_no_context}'")
        answer_no_context = await retrieval_agent_for_tool_test.answer_from_knowledge_base(user_query=test_query_no_context)
        print(f"\nAnswer from tool (no context): {answer_no_context}")

    if __name__ == '__main__':
        asyncio.run(test_retrieval_agent_tool())
