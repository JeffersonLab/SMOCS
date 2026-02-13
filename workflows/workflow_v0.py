import os
import asyncio
import argparse
from typing import Annotated, List, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages        # Instead of creating a wrapper tool node around the ToolNode to manually append the ToolMessage to state['messages']
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import ToolNode



def parse_args(args: dict = {}) -> dict:
    # Returns dict that has info from args (higher priority) and parsed args (lower priority)
    arg_parser = argparse.ArgumentParser(description='Agent subscribing to MCP server with a human-in-the-loop (HITL) before every tool call.')
    arg_parser.add_argument('--llm_name', type=str, default='qwen2.5:14b', help='Name of Ollama LLM to use.')       # 'llama3.2:3b', 'llama3.1:8b', 'mistral-small', 'qwen2.5:14b'
    arg_parser.add_argument('--mcp_server_filepath', type=str, default=os.path.join(os.path.dirname(__file__), '../MCP/server_v0.py'), help='Path to MCP server python file defining the tools the LLM will use with stdio transport.')
    arg_parser.add_argument('--workflow_diagram_savepath', type=str, default=None, help='If provided, a visual representation of the agentic workflow will be saved to the provided path.')
    parsed_args = arg_parser.parse_args()
    parsed_args = vars(parsed_args)
    args.update(parsed_args)
    return args



class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]



async def get_agent(llm_name: str, mcp_server_filepath: str = None):
    async def get_tools():
        mcp_client = MultiServerMCPClient({
            'server_1': {
                'command': 'python',
                #'args': [os.path.join(os.path.dirname(__file__), '../MCP/server.py')],
                'args': [mcp_server_filepath],
                'transport': 'stdio',
            }
        })
        tools = await mcp_client.get_tools()
        return tools
    

    async def llm_call(state: AgentState) -> dict:
        """Call LLM and return new message"""
        response = await llm_with_tools.ainvoke(state['messages'])
        return {"messages": [response]}


    async def human_approval(state: AgentState) -> dict:
        """Ask human for approval before executing tools"""
        last_message = state['messages'][-1]
        print("AI wants to use the following tool(s):")
        for i, tool_call in enumerate(last_message.tool_calls, 1):
            print(f"{i}. Tool: {tool_call['name']} | Args: {tool_call['args']}")
        approval = input("Allow tool execution? (yes/no): ").strip().lower()
        if approval in {'yes', 'y'}:
            # Proceed with tool execution - return empty dict to not modify state
            return {}
        else:
            # Block tool execution - ask for feedback
            feedback = input("Please provide feedback or a new instruction: ").strip()
            # Create a fake ToolMessage indicating the tool was blocked
            blocked_messages = []
            for tool_call in last_message.tool_calls:
                blocked_messages.append(
                    ToolMessage(
                        content=f"Tool execution blocked by user. User feedback: {feedback}",
                        tool_call_id=tool_call['id']
                    )
                )
            return {"messages": blocked_messages}


    def route_after_llm(state: AgentState) -> str:
        """Route to approval/tools or end based on tool calls"""
        last_message = state['messages'][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return 'human_approval'
        return 'end'
    

    def route_after_approval(state: AgentState) -> str:
        """Route to tools or llm based on approval"""
        last_message = state['messages'][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            # If last message is AIMessage with tool_calls, approval was given as the human_approval node does NOT add any message in this case
            return 'tools'
        else:
            # human_approval does append its blocked messages that are NOT of type AIMessage
            return 'llm_call'


    # Setup
    tools = await get_tools()
    llm = ChatOllama(
        model=llm_name,
        temperature=0.0,
    )
    llm_with_tools = llm.bind_tools(tools)

    # Build graph
    graph = StateGraph(AgentState)
    graph.add_node('llm_call', llm_call)
    graph.add_node('human_approval', human_approval)
    graph.add_node('tools', ToolNode(tools=tools))
    graph.add_edge(START, 'llm_call')
    # After LLM, check if tools needed
    graph.add_conditional_edges(
        'llm_call',
        route_after_llm,
        {'human_approval': 'human_approval', 'end': END}
    )
    # After human approval, execute tools or go back to LLM
    graph.add_conditional_edges(
        'human_approval',
        route_after_approval,
        {'tools': 'tools', 'llm_call': 'llm_call'}
    )
    # After tools, go back to LLM
    graph.add_edge('tools', 'llm_call')
    
    agent = graph.compile()
    return agent


async def main(args: dict):
    """Main async function to run the chat loop"""
    llm_name = args['llm_name']
    mcp_server_filepath = args['mcp_server_filepath']
    workflow_diagram_savepath = args['workflow_diagram_savepath']

    agent = await get_agent(llm_name, mcp_server_filepath)

    if workflow_diagram_savepath is not None:
        png_data = agent.get_graph().draw_mermaid_png()
        os.makedirs(os.path.dirname(workflow_diagram_savepath), exist_ok=True)
        with open(workflow_diagram_savepath, 'wb') as file:
            file.write(png_data)

    state: AgentState = {"messages": []}
    while True:
        user_input = input('Human: ')
        if user_input.lower() == 'exit':
            break
        elif user_input.lower() == 'clear':
            state['messages'] = []
            print('Cleared conversation history')
            continue
        
        # Append new human message to existing state
        state = await agent.ainvoke(
            {"messages": state['messages'] + [HumanMessage(content=user_input)]}
        )
        print('AI:', state['messages'][-1].content)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))

# Example Query: List down all the "PVs" in the "epics" service in the "./orchestration/config.yaml" file.