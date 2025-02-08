import os
import subprocess
import asyncio
from typing import Optional
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import ExternalTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient


def execute_shell_command(command: str) -> str:
    """Execute shell command and return output"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            text=True,
            capture_output=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error executing command: {e.stderr}"

def create_model_client() -> OpenAIChatCompletionClient:
    """Create configuration for the agents."""
    return OpenAIChatCompletionClient(
        model="ollama/phi4",
        api_key="NotRequiredSinceWeAreLocal",
        base_url="http://localhost:4000",
        model_capabilities={
            "json_output": False,
            "vision": False,
            "function_calling": False,
        }
    )

def create_agents(model_client: OpenAIChatCompletionClient) -> tuple[AssistantAgent, UserProxyAgent]:
    shell_expert = AssistantAgent(
        name="shell_expert",
        system_message="""You are a Linux shell expert. Convert user requests into appropriate shell commands. 
        Only provide the shell commands, no explanations. Commands must be safe and non-destructive.
        For example, user request is "show content of folder ~/", then return "ls -lha ~/".
        If a request could be dangerous, respond with 'UNSAFE COMMAND REQUEST'.
        Use the command output and conversation history to provide better commands.""",
        model_client=model_client,
    )

    user_proxy = UserProxyAgent(
        name="user_proxy"
    )
    
    return shell_expert, user_proxy

async def process_command_stream(stream) -> Optional[str]:
    try:
        async for event in stream:
            if hasattr(event, 'content'):
                return event.content
    except asyncio.CancelledError:
        return None
    return None

async def main():
    model_client = create_model_client()
    shell_expert, user_proxy = create_agents(model_client)
    
    # Create external termination condition
    termination = ExternalTermination()
    
    team = RoundRobinGroupChat(
        participants=[shell_expert, user_proxy],
        max_turns=1,
        termination_condition=termination
    )

    print("Shell Command Assistant (Type 'exit' to quit)")
    print("-" * 50)

    while True:
        user_input = input("\nEnter your request: ").strip()
        if user_input.lower() == 'exit':
            break

        try:
            # Reset termination for new request
            termination.set()
            # wait for the team to finish
            await team.run(task=user_input)
            
            stream = team.run_stream(task=user_input)
            command = await process_command_stream(stream)

            if not command:
                print("No command generated.")
                continue

            if command == "UNSAFE COMMAND REQUEST":
                print("⚠️ This command request was deemed unsafe and will not be executed.")
                continue

            print("\nGenerated command:")
            print("-" * 50)
            print(command)
            print("-" * 50)

            if input("Execute this command? (y/n): ").strip().lower() != 'y':
                print("Command execution cancelled.")
                continue

            print("\nOutput:")
            print("-" * 50)
            result = execute_shell_command(command)
            print(result)

        except Exception as e:
            print(f"Error processing request: {str(e)}")
            # Set termination in case of error
            await termination.set()

if __name__ == "__main__":
    asyncio.run(main()) 