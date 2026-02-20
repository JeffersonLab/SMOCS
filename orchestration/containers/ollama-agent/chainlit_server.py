import os
from typing import Annotated, List, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages        # Instead of creating a wrapper tool node around the ToolNode to manually append the ToolMessage to state['messages']
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import ToolNode
import chainlit as cl



class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]



async def get_agent():
    async def get_tools():
        mcp_client = MultiServerMCPClient({
            'server_1': {
                'command': 'python',
                'args': [os.path.join(os.path.dirname(__file__), 'mcp_server.py')],
                'transport': 'stdio',
            }
        })
        tools = await mcp_client.get_tools()
        return tools
    

    async def llm_call(state: AgentState) -> dict:
        """Call LLM and return new message"""
        response = await llm_with_tools.ainvoke(state['messages'])
        return {'messages': [response]}


    async def human_approval(state: AgentState) -> dict:
        """Ask human for approval before executing tools using Chainlit UI"""
        last_message = state['messages'][-1]
        
        # Format tool calls for display
        tool_descriptions = []
        for i, tool_call in enumerate(last_message.tool_calls, 1):
            tool_descriptions.append(
                f"**{i}. Tool:** `{tool_call['name']}`\n**Args:** `{tool_call['args']}`"
            )
        
        tool_text = "\n\n".join(tool_descriptions)
        
        # Use Chainlit's AskActionMessage for approval
        res = await cl.AskActionMessage(
            content=f'AI wants to use the following tool(s):\n\n{tool_text}\n\nDo you approve?',
            actions=[
                cl.Action(name='approve', label='✅ Approve', payload={'decision': 'yes'}),
                cl.Action(name='reject', label='❌ Reject', payload={'decision': 'no'}),
            ],
        ).send()
        
        if res and res.get('payload', {}).get('decision') == 'yes':
            # Approved - return empty dict to proceed
            return {}
        else:
            # Rejected - ask for feedback
            feedback_res = await cl.AskUserMessage(
                content='Tool execution blocked. Please provide feedback or a new instruction:',
                timeout=300,  # 5 minutes timeout
            ).send()
            
            feedback = feedback_res['output'] if feedback_res else "No feedback provided"
            
            # Create blocked tool messages
            blocked_messages = []
            for tool_call in last_message.tool_calls:
                blocked_messages.append(
                    ToolMessage(
                        content=f'Tool execution blocked by user. User feedback: {feedback}',
                        tool_call_id=tool_call['id']
                    )
                )
            return {'messages': blocked_messages}


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
            # If last message is AIMessage with tool_calls, approval was given as the human_approval node does NOT add any message
            return 'tools'
        return 'llm_call'


    # Setup
    tools = await get_tools()
    llm = ChatOllama(model=os.environ['LLM_NAME'], temperature=0.0)
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


async def stream_msg(msg: cl.Message, content: str) -> None:
    start_idx = 0
    while start_idx < len(content):
        end_idx = start_idx + int(os.environ.get('STREAM_CHUNK_SIZE', 1))
        await msg.stream_token(content[start_idx : end_idx])
        await cl.sleep(float(os.environ.get('STREAM_SLEEP_TIME', 0.02)))
        start_idx = end_idx


async def stream_content(content: str) -> cl.Message:
    msg = cl.Message(content='')
    await msg.send()    # create message bubble in the UI before streaming
    await stream_msg(msg, content)
    await msg.update()
    return msg


@cl.on_chat_start
async def on_chat_start() -> None:
    """
    Called when a new chat session starts.
    Initialize the agent and store it in the user session.
    """
    await stream_content('Hi, please wait while agent is being initialized ...')
    agent = await get_agent()
    try:
        png_data = agent.get_graph().draw_mermaid_png()
        with open(os.path.join(os.path.dirname(__file__), 'diagram.png'), 'wb') as file:
            file.write(png_data)
    except Exception as e:
        print(f'An error occurred while attempting to save "diagram.png" representation of the workflow to disk: {e}') 
        print('Skipping saving "diagram.png" ...')
    cl.user_session.set('agent', agent)
    cl.user_session.set('state', {'messages': []})
    await stream_content(f'''Agent "{os.environ['LLM_NAME']}" initialized successfully. How may I assist you today?''')


@cl.on_message
async def on_message(message: cl.Message) -> None:
    if message.content.strip().lower() == 'clear':
        cl.user_session.set('state', {'messages': []})
        await stream_content('Chat history has been cleared. How may I assist you today?')
    else:
        agent = cl.user_session.get('agent')
        state = cl.user_session.get('state')

        messages = state['messages'] + [HumanMessage(content=message.content)]
        max_messages = int(os.environ.get('MAX_MESSAGES', len(messages)))
        messages = messages[-max_messages :]

        msg = cl.Message(content='')
        await msg.send()
        num_printed_chars = 0
        latest_state = None
        async for latest_state in agent.astream({'messages': messages}, stream_mode='values'):
            last_msg = latest_state['messages'][-1]
            if isinstance(last_msg, AIMessage) and last_msg.content:
                delta = last_msg.content[num_printed_chars :]
                if delta:
                    await stream_msg(msg, delta)
                    num_printed_chars += len(delta)
        await msg.update()
        if latest_state is not None:
            cl.user_session.set('state', {'messages': latest_state['messages']})

    

# Example Query: List down all the "PVs" in the "epics" service in the "./orchestration/config.yaml" file.