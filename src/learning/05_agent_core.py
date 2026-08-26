"""
Step 5: Agent core -- Ollama (local LLM) + tool calling + LangGraph

PREREQUISITE (not pip -- a separate local application):
1. Install Ollama for Windows: https://ollama.com/download
2. Check what you already have (you've used qwen3:8b before for SkillRadar):
       ollama list
   If a Qwen model is already there, you can skip pulling a new one and
   just set MODEL_NAME below to match it.
3. Otherwise pull one: ollama pull qwen2.5:7b
   (if that's slow or your machine struggles, try qwen2.5:3b instead --
   both support tool calling)
4. Ollama runs as a background service on http://localhost:11434
   automatically once installed -- nothing else to start manually.

Install (pip): pip install langgraph langchain-ollama
"""

import datetime
import time
from typing import Annotated
from typing_extensions import TypedDict

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Change this to whatever model you actually have pulled -- run `ollama list`
# to check first before downloading anything new.
MODEL_NAME = "qwen2.5:3b"


# --- Define at least one real tool so tool-calling has something to do ---
@tool
def get_current_time() -> str:
    """Returns the current date and time. Use this whenever the user asks
    what time or day it is."""
    return datetime.datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")


tools = [get_current_time]

# temperature=0 -- deterministic-ish output. For a voice agent you want
# consistent, predictable behavior, not creative variation.
llm = ChatOllama(model=MODEL_NAME, temperature=0)

# bind_tools tells the model WHAT tools exist and their descriptions/schemas
# (pulled from each @tool function's docstring and type hints) -- this is
# how the model knows get_current_time exists and when to use it.
llm_with_tools = llm.bind_tools(tools)


# --- The graph's shared state: just a running list of conversation messages ---
class AgentState(TypedDict):
    # add_messages is a special reducer: instead of REPLACING the messages
    # list on every node update, it APPENDS to it -- this is what lets
    # conversation history accumulate correctly across the graph's steps.
    messages: Annotated[list, add_messages]


def call_model(state: AgentState):
    """The 'brain' node: sends the full message history to the LLM, gets
    back either a normal text reply OR a request to call a tool."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState):
    """Routing logic: inspect the model's last response. If it asked to
    call a tool, route to the tools node. Otherwise, we're done."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


# --- Wire the graph together ---
graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", ToolNode(tools))  # prebuilt: runs whichever tool the model asked for

graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")  # after running a tool, ALWAYS go back to the model with the result

agent_graph = graph.compile()


if __name__ == "__main__":
    print(f"Agent ready (model: {MODEL_NAME}). Type a message, or 'quit' to exit.\n")
    print("Try asking: 'what time is it?' to see tool-calling in action.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ("quit", "exit"):
            break

        t0 = time.time()
        result = agent_graph.invoke({"messages": [("user", user_input)]})
        elapsed_ms = round((time.time() - t0) * 1000)
        final_message = result["messages"][-1]
        print(f"Agent: {final_message.content}  ({elapsed_ms}ms)\n")