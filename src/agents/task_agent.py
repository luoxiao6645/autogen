import os
import yagmail # Added for real email sending
from typing import Optional, Annotated
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelClient
from autogen_core.tools import tool

# Load environment variables from .env file
load_dotenv()

class TaskExecutionAgent(AssistantAgent):
    def __init__(
        self,
        name: str,
        model_client: ModelClient,
        system_message: Optional[str] = None,
        **kwargs
    ):
        DEFAULT_SYSTEM_MESSAGE = (
            "You are a specialized AI assistant for task execution. "
            "You are provided with tools to perform specific actions such as sending emails. "
            "When requested to perform a task, identify the correct tool and use it with the provided arguments. "
            "Report the outcome of the tool execution (success or failure, and any relevant output from the tool)."
        )

        super().__init__(
            name,
            model_client=model_client,
            system_message=system_message or DEFAULT_SYSTEM_MESSAGE,
            **kwargs
        )
        # Register its own tools
        self.register_tools([self.send_email]) # Renamed from send_mock_email

    @tool()
    def send_email(
        self,
        recipient: Annotated[str, "The email address of the recipient."],
        subject: Annotated[str, "The subject of the email."],
        body: Annotated[str, "The body content of the email."]
    ) -> str:
        """
        Sends an email to the specified recipient with the given subject and body.
        Requires email configuration (sender address, password, SMTP server/port) to be set in environment variables.
        """
        print(f"[TaskExecutionAgent - Real Email Tool Called]")
        print(f"  Attempting to send email to: {recipient}")
        print(f"  Subject: {subject}")
        # print(f"  Body: \"{body}\"") # Potentially long, maybe log selectively

        sender_address = os.getenv("EMAIL_SENDER_ADDRESS")
        sender_password = os.getenv("EMAIL_SENDER_PASSWORD")
        smtp_server = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com") # Default to Gmail if not set
        smtp_port_str = os.getenv("EMAIL_SMTP_PORT", "587") # Default to 587 if not set

        if not all([sender_address, sender_password]):
            error_msg = "Email configuration incomplete. Please set EMAIL_SENDER_ADDRESS and EMAIL_SENDER_PASSWORD in environment variables."
            print(f"  Error: {error_msg}")
            return error_msg

        try:
            smtp_port = int(smtp_port_str)
        except ValueError:
            error_msg = f"Invalid EMAIL_SMTP_PORT: '{smtp_port_str}'. Must be an integer."
            print(f"  Error: {error_msg}")
            return error_msg

        try:
            yag = yagmail.SMTP(sender_address, sender_password, host=smtp_server, port=smtp_port)
            yag.send(
                to=recipient,
                subject=subject,
                contents=body
            )
            success_msg = f"Email successfully sent to {recipient} with subject '{subject}'."
            print(f"  Success: {success_msg}")
            return success_msg
        except yagmail.errors.YagConnectionClosed as e:
            error_msg = f"Failed to send email: Connection error. Check SMTP server/port and network. Details: {str(e)}"
            print(f"  Error: {error_msg}")
            return error_msg
        except yagmail.errors.YagSMTPAuthenticationError as e:
            error_msg = f"Failed to send email: Authentication error. Check sender email/password. Details: {str(e)}"
            print(f"  Error: {error_msg}")
            return error_msg
        except Exception as e:
            # Catch any other yagmail or general exceptions
            error_msg = f"Failed to send email due to an unexpected error: {str(e)}"
            print(f"  Error: {error_msg}")
            return error_msg

if __name__ == '__main__':
    import os
    import asyncio
    from dotenv import load_dotenv
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    from autogen_agentchat.agents import UserProxyAgent

    load_dotenv() # Ensure .env is loaded for testing this file directly
    api_key = os.getenv("OPENAI_API_KEY")

    # --- Test Real Email Sending (Requires .env to be configured with email credentials) ---
    async def test_real_email_tool():
        print("\n--- Testing Real Email Tool ---")
        if not os.getenv("EMAIL_SENDER_ADDRESS") or not os.getenv("EMAIL_SENDER_PASSWORD"):
            print("EMAIL_SENDER_ADDRESS or EMAIL_SENDER_PASSWORD not set in .env. Skipping real email test.")
            print("To test, configure these in .env, along with SMTP server/port if not Gmail defaults.")
            return

        # This test directly calls the tool method, not via LLM tool_call generation.
        # For LLM tool_call, the agent's LLM would need to be prompted appropriately.
        task_agent_for_direct_test = TaskExecutionAgent(
            name="DirectEmailTester",
            model_client=OpenAIChatCompletionClient(model="gpt-3.5-turbo", api_key=api_key) # LLM needed for agent init
        )

        test_recipient = os.getenv("EMAIL_TEST_RECIPIENT", os.getenv("EMAIL_SENDER_ADDRESS")) # Send to self if no other recipient
        if not test_recipient:
            print("No test recipient found (EMAIL_TEST_RECIPIENT or EMAIL_SENDER_ADDRESS). Skipping direct tool call test.")
            return

        print(f"Attempting to send a real test email to: {test_recipient}")
        result = task_agent_for_direct_test.send_email(
            recipient=test_recipient,
            subject="AutoGen Real Email Test",
            body="This is a test email sent by the AutoGen TaskExecutionAgent using yagmail."
        )
        print(f"Result of real email send attempt: {result}")

    # --- Test LLM-driven tool call (similar to previous test) ---
    async def test_llm_driven_email_tool_call():
        print("\n--- Testing LLM-Driven Email Tool Call ---")
        if not api_key:
            print("OPENAI_API_KEY not found.")
            return
        if not os.getenv("EMAIL_SENDER_ADDRESS") or not os.getenv("EMAIL_SENDER_PASSWORD"):
            print("Email credentials not set in .env. LLM might not be able to successfully use the tool if it tries.")
            # The tool will return an error, which is a valid test of its internal error handling.

        llm_for_caller = OpenAIChatCompletionClient(model="gpt-4o", api_key=api_key) # Use GPT-4o for better tool use
        llm_for_task_agent = OpenAIChatCompletionClient(model="gpt-3.5-turbo", api_key=api_key)

        task_agent_for_llm_test = TaskExecutionAgent(
            name="TaskExecutorLLM",
            model_client=llm_for_task_agent
        )

        caller_agent = UserProxyAgent(
            name="EmailRequesterProxy",
            human_input_mode="NEVER",
            model_client=llm_for_caller,
            system_message="You need to send an email. Use the 'send_email' tool. Provide all arguments clearly."
        )

        caller_agent.register_tools(tools=[task_agent_for_llm_test.send_email])

        test_recipient_for_llm = os.getenv("EMAIL_TEST_RECIPIENT", os.getenv("EMAIL_SENDER_ADDRESS"))
        if not test_recipient_for_llm:
            test_recipient_for_llm = "test@example.com" # Fallback if no env var for LLM test
            print(f"Using fallback recipient for LLM test: {test_recipient_for_llm}")


        print(f"Initiating LLM-driven chat to trigger 'send_email' tool call to {test_recipient_for_llm}...")
        await caller_agent.a_initiate_chat(
            recipient=task_agent_for_llm_test,
            message=f"Please send an email to '{test_recipient_for_llm}' with subject 'LLM Test Email' and body 'This email was triggered by an LLM making a tool call via AutoGen.'",
            max_turns=3
        )

        history = caller_agent.chat_messages_for_summary(task_agent_for_llm_test)
        print("\nLLM-driven Chat History:")
        for msg in history:
            print(f"- Role: {msg.get('role')}, Name: {msg.get('name')}, Content: {msg.get('content')}")
            if msg.get("tool_calls"):
                print(f"  Tool Calls: {msg.get('tool_calls')}")
            if msg.get("tool_responses"):
                 print(f"  Tool Responses: {msg.get('tool_responses')}")


    if __name__ == '__main__':
        # Run the LLM-driven test first as it also tests error handling if creds aren't set
        asyncio.run(test_llm_driven_email_tool_call())
        # Then attempt direct real email if configured
        asyncio.run(test_real_email_tool())
