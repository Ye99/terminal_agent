import os
from typing import Annotated, Dict, TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import Graph, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
import subprocess
from typing import List, Tuple, Optional
from IPython.display import Image, display
import logging
from rich.console import Console
from rich.panel import Panel
from rich.logging import RichHandler

class AgentState(TypedDict):
    messages: List[HumanMessage | AIMessage]
    command: Optional[str]
    output: Optional[str]
    next: str

def execute_shell_command(command: str) -> str:
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

def create_linux_expert():
    llm = Ollama(
        model="phi4",
        base_url="http://localhost:11434",
        temperature=0
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Linux shell expert. Convert user requests into appropriate shell commands.
        Only provide the shell commands, no explanations. Commands must be safe and non-destructive.
        Do not surrund your response with ``` or ```bash. just return the command, for instance "ls -l /"
        If a request could be dangerous, respond with 'UNSAFE COMMAND REQUEST'."""),
        ("human", "{input}")
    ])
    
    return prompt | llm | StrOutputParser()

def create_linux_reviewer():
    llm = Ollama(
        model="phi4",
        base_url="http://localhost:11434",
        temperature=0
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Linux command reviewer. Review the proposed commands for safety and correctness.
        If the command is safe and correct, respond with 'APPROVED' followed by your reasoning.
        If multiple commands are provided, review each one and respond with 'APPROVED' for the best one.
        Do not surrund your suggest command with ``` or ```bash. just the command, for instance "ls -l /"
        If none are safe, explain what needs to be fixed."""),
        ("human", "{input}")
    ])
    
    return prompt | llm | StrOutputParser()

# Set up rich console and logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("shell_commander")

def log_state_transition(from_node: str, to_node: str, state: AgentState):
    console.print(Panel(
        f"[yellow]Transition:[/yellow] {from_node} -> {to_node}\n"
        f"[blue]Command:[/blue] {state['command']}\n"
        f"[green]Messages:[/green] {len(state['messages'])}\n"
        f"[cyan]Next:[/cyan] {state['next']}"
    ))

def user_input(state: AgentState) -> AgentState:
    messages = state["messages"]
    if not messages:  # First interaction
        user_msg = input("\nEnter your request: ").strip()
        if user_msg.lower() == 'quit()':
            logger.info("Quit command received")
            state["next"] = "end"
            state["command"] = "quit"
            return state
        messages.append(HumanMessage(content=user_msg))
        logger.info(f"User Input: {user_msg}")
    state["next"] = "linux_expert"
    return state

def linux_expert(state: AgentState) -> AgentState:
    logger.info("Linux Expert: Generating command...")
    expert = create_linux_expert()
    messages_text = "\n".join([msg.content for msg in state["messages"]])
    response = expert.invoke({"input": messages_text})
    state["command"] = response
    state["next"] = "linux_reviewer"
    logger.info(f"Generated Command: {response}")
    return state

def linux_reviewer(state: AgentState) -> AgentState:
    logger.info("Linux Reviewer: Reviewing command...")
    reviewer = create_linux_reviewer()
    review = reviewer.invoke({"input": f"Review this command: {state['command']}"})
    logger.info(f"Review Result: {review}")
    
    if "APPROVED" in review.strip():
        # If there are multiple commands, pick the first approved one
        commands = state["command"].strip().split('\n')
        if len(commands) > 1:
            logger.info("Multiple commands found, selecting first one")
            state["command"] = commands[0]
            logger.info(f"Selected command: {state['command']}")
        
        state["next"] = "user_approval"
    else:
        feedback = f"Command needs revision: {review}"
        state["messages"].append(AIMessage(content=feedback))
        state["next"] = "linux_expert"
        logger.warning(f"Command rejected: {review}")
    
    return state

def user_approval(state: AgentState) -> AgentState:
    print("\nGenerated command:")
    print("-" * 50)
    print(state["command"])
    print("-" * 50)
    
    if input("Execute this command? (y/n): ").strip().lower() != 'y':
        print("Command execution cancelled.")
        # Clear messages to start fresh with new input
        state["messages"] = []
        state["next"] = "user_input"
        return state
    
    print("\nOutput:")
    print("-" * 50)
    output = execute_shell_command(state["command"])
    print(output)
    state["output"] = output
    
    if input("\nDo you want to analyze the output? (y/n): ").strip().lower() == 'y':
        state["messages"].append(AIMessage(content=f"Command output: {output}"))
        state["next"] = "linux_expert"
    else:
        # Clear messages to start fresh with new input
        state["messages"] = []
        state["next"] = "user_input"
    return state

def end_node(state: AgentState) -> AgentState:
    logger.info("Ending program")
    return state

def create_workflow() -> Graph:
    workflow = StateGraph(AgentState)
    
    # Add nodes with instrumentation
    def create_instrumented_node(name: str, func):
        def instrumented(state: AgentState) -> AgentState:
            logger.info(f"Entering {name}")
            result = func(state)
            log_state_transition(name, result["next"], result)
            return result
        return instrumented

    workflow.add_node("user_input", create_instrumented_node("user_input", user_input))
    workflow.add_node("linux_expert", create_instrumented_node("linux_expert", linux_expert))
    workflow.add_node("linux_reviewer", create_instrumented_node("linux_reviewer", linux_reviewer))
    workflow.add_node("user_approval", create_instrumented_node("user_approval", user_approval))
    workflow.add_node("end", end_node)  # Add end node
    
    # Set entry point
    workflow.set_entry_point("user_input")
    
    # Define edges with clear flow
    workflow.add_edge("user_input", "linux_expert")
    workflow.add_edge("linux_expert", "linux_reviewer")
    
    # Add conditional routing for user_input to handle exit
    workflow.add_conditional_edges(
        "user_input",
        lambda x: x["next"],
        {
            "linux_expert": "linux_expert",
            "end": "end"  # Add direct path to end
        }
    )
    
    # Add conditional routing
    workflow.add_conditional_edges(
        "linux_reviewer",
        lambda x: x["next"],
        {
            "user_approval": "user_approval",
            "linux_expert": "linux_expert"
        }
    )
    
    # Add conditional routing for user_approval
    workflow.add_conditional_edges(
        "user_approval",
        lambda x: x["next"],
        {
            "user_input": "user_input",
            "linux_expert": "linux_expert"
        }
    )

    graph = workflow.compile() 
    # display(Image(graph.get_graph().draw_mermaid_png()))
   
    return graph

def main():
    console.print("[bold green]Shell Command Assistant[/bold green]")
    console.print("[bold]Type 'quit()' to quit[/bold]")
    console.print("-" * 50)

    workflow = create_workflow()
    
    config = {
        "messages": [],
        "command": None,
        "output": None,
        "next": "user_input"
    }
    
    try:
        for output in workflow.stream(config):
            if output.get("command") == "quit" or output.get("next") == "end":
                logger.info("Exiting program")
                break
    except Exception as e:
        logger.error(f"Workflow error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()