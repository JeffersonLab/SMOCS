import os
import time
import yaml
import myers
from typing import Annotated, List, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages        # Instead of creating a wrapper tool node around the ToolNode to manually append the ToolMessage to state['messages']
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import ToolNode
import chainlit as cl



class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]



async def get_agent(tools: list):
    """
    Build and return the LangGraph agent.
    Tools are passed in from the already-alive MCP client.
    """

    async def llm_call(state: AgentState) -> dict:
        """Call LLM and return new message"""
        response = await llm.ainvoke(state['messages'])
        return {'messages': [response]}
    

    def route_after_llm(state: AgentState) -> str:
        """Route to human_approval or end based on whether tool calls exist"""
        last_message = state['messages'][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return 'human_approval'
        return 'end'


    async def human_approval(state: AgentState) -> dict:
        """Ask human for approval before executing tools using Chainlit UI"""
        last_message = state['messages'][-1]
        
        # Format tool calls for display
        tool_descriptions = []
        for i, tool_call in enumerate(last_message.tool_calls, 1):
            if tool_call['name'] == 'write_yaml':
                filepath = tool_call['args']['path']
                val = tool_call['args']['val']
                val_str = yaml.safe_dump(val, sort_keys=True, indent=4, default_flow_style=False)
                if os.path.isfile(filepath):
                    # Apply myers algortithm to determine the shortest path of changes from old_val_str to val_str
                    # NOTE: difflib.ndiff did NOT work well when there are common sections. It shows the changes in unexpected places.
                    with open(filepath, 'r') as file:
                        old_val = yaml.safe_load(file)
                    old_val_str = yaml.safe_dump(old_val, sort_keys=True, indent=4, default_flow_style=False)
                    # a) Single file with "+/-" to indicate insertions/removals
                    diff_result = myers.diff(old_val_str.splitlines(), val_str.splitlines())
                    final_output = []
                    for action, line in diff_result:
                        if action == 'k':
                            # keep
                            final_output.append(f"  {line}")  # Two spaces for alignment
                        elif action == 'r':
                            # remove
                            final_output.append(f"- {line}")  # Minus for deletions
                        elif action == 'i':
                            # insert
                            final_output.append(f"+ {line}")  # Plus for additions
                        else:
                            # omit
                            assert action == 'o', f'actions can be one of KEEP/REMOVE/INSERT/OMIT (k/r/i/o) !!!'
                            raise NameError('Undefined behavior with OMIT "o" action !!!')
                    final_output: str = '\n'.join(final_output)
                    # b) TODO: More fancy side-by-side view like vscode
                    txt = f"### Tool {i}: `write_yaml`\n**📝 Overwriting File:** `{filepath}`\n```diff\n{final_output}\n```"
                else:
                    txt = f"### Tool {i}: `write_yaml`\n**📝 File:** `{filepath}`\n```diff\n{val_str}\n```"
            else:
                txt = f"### Tool {i}: `{tool_call['name']}`\n**Args:** `{tool_call['args']}`"
            tool_descriptions.append(txt)
        tool_text = "\n\n".join(tool_descriptions)
        
        # Use Chainlit's AskActionMessage for approval
        res = await cl.AskActionMessage(
            content=f'AI wants to use the following tool(s):\n\n{tool_text}\n\nDo you approve?',
            actions=[
                cl.Action(name='approve', label='✅ Approve', payload={'decision': 'yes'}),
                cl.Action(name='reject', label='❌ Reject', payload={'decision': 'no'}),
            ],
            timeout=600,  # 10 minutes timeout
        ).send()
        
        if res and res.get('payload', {}).get('decision') == 'yes':
            # Approved — return empty dict so the AIMessage with tool_calls remains last
            return {}
        else:
            # Rejected - ask for feedback
            feedback_res = await cl.AskUserMessage(
                content='Tool execution blocked. Please provide feedback or a new instruction:',
                timeout=300,  # 5 minutes timeout
            ).send()
            feedback = feedback_res['output'] if feedback_res else "No feedback provided"
            return {'messages': [HumanMessage(content=feedback)]}
    

    def route_after_approval(state: AgentState) -> str:
        """
        Route to tools or llm_call based on approval outcome.
        - Approved ==> last message is still the AIMessage with tool_calls as the human_approval node does NOT add any message ==> 'tools'
        - Rejected ==> last message is NOT an AIMessage (blocked result) ==> 'llm_call'
        """
        last_message = state['messages'][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            # If last message is AIMessage with tool_calls, approval was given as the human_approval node does NOT add any message
            return 'tools'
        return 'llm_call'


    # Setup
    llm = ChatOllama(model=os.environ['LLM_NAME'], temperature=0.0).bind_tools(tools)

    # Build graph
    graph = StateGraph(AgentState)
    graph.add_node('llm_call', llm_call)
    graph.add_node('human_approval', human_approval)
    graph.add_node('tools', ToolNode(tools=tools, messages_key='messages'))
    graph.add_edge(START, 'llm_call')
    graph.add_conditional_edges(
        'llm_call',
        route_after_llm,
        {'human_approval': 'human_approval', 'end': END}
    )
    graph.add_conditional_edges(
        'human_approval',
        route_after_approval,
        {'tools': 'tools', 'llm_call': 'llm_call'}
    )
    graph.add_edge('tools', 'llm_call')
    agent = graph.compile()
    return agent


async def stream_msg(msg: cl.Message, content: str) -> None:
    start_idx = 0
    while start_idx < len(content):
        end_idx = start_idx + int(os.environ.get('STREAM_CHUNK_SIZE', 1))
        await msg.stream_token(content[start_idx : end_idx])
        await cl.sleep(float(os.environ.get('STREAM_SLEEP_TIME', 0.00)))
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

    mcp_client_configs = {
        'server_1': {
            'command': 'python',
            'args': [os.path.join(os.path.dirname(__file__), 'mcp_server.py')],
            'transport': 'stdio',
        }
    }
    mcp_client = MultiServerMCPClient(mcp_client_configs)
    
    tools = []
    mcp_contexts = []
    for mcp_server_name in mcp_client_configs.keys():
        context = mcp_client.session(mcp_server_name)
        # Enter the context (this starts the subprocess and stays open)
        session = await context.__aenter__()
        server_tools = await load_mcp_tools(session)
        tools.extend(server_tools)
        mcp_contexts.append(context)

    agent = await get_agent(tools)

    try:
        png_data = agent.get_graph().draw_mermaid_png()
        with open(os.path.join(os.path.dirname(__file__), 'diagram.png'), 'wb') as file:
            file.write(png_data)
    except Exception as e:
        print(f'An error occurred while attempting to save "diagram.png" representation of the workflow to disk: {e}') 
        print('Skipping saving "diagram.png" ...')
    cl.user_session.set('agent', agent)
    cl.user_session.set('mcp_contexts', mcp_contexts)
    cl.user_session.set('state', {'messages': []})
    await stream_content(f'''Agent "{os.environ['LLM_NAME']}" initialized successfully. How may I assist you today?''')


@cl.on_chat_end
async def on_chat_end() -> None:
    mcp_contexts = cl.user_session.get('mcp_contexts')
    for mcp_context in mcp_contexts:
        try:
            await mcp_context.__aexit__(None, None, None)
        except RuntimeError as e:
            print(f'Warning: MCP context cleanup error (safe to ignore): {e}')
        except Exception as e:
            print(f'Warning: Unexpected MCP context cleanup error: {e}')


@cl.on_message
async def on_message(message: cl.Message) -> None:
    if message.content.strip().lower() == 'clear':
        cl.user_session.set('state', {'messages': []})
        await stream_content('Chat history has been cleared. How may I assist you today?')
    else:
        agent = cl.user_session.get('agent')
        state = cl.user_session.get('state')
        state['messages'].append(HumanMessage(content=message.content))
        t_0 = time.perf_counter()
        async for update_dict in agent.astream(state, stream_mode='updates'):
            update_time = time.perf_counter() - t_0
            nodes_names = list(update_dict.keys())      # Only one item in update_dict since there are no parallel branches in agent.
            await stream_content(f'Nodes "{nodes_names}" finished in {update_time:.2f} sec')
            for state_delta in update_dict.values():
                # state_delta has what is returned from the nodes, not the aggregated full state
                if isinstance(state_delta, dict) and ('messages' in state_delta):
                    new_msgs = state_delta['messages']
                    state['messages'].extend(new_msgs)
                    for msg in new_msgs:
                        if isinstance(msg, AIMessage) and msg.content:
                            await stream_content(msg.content)
            t_0 = time.perf_counter()
        cl.user_session.set('state', state)


        
    
'''
Queries
-------
a)
List down all the "PVs" in the "epics" service in the "./orchestration/config.yaml" file.
b)
Which agent uses the first two only?
c)
Add the following to the configuration file:
1. Append a new PV called "IPMK501.XPOS" to the list of epics PVs
2. Add a new autoencoder called "autoencoder_agent3" that is identical to "autoencoder_agent2". The only difference is that it uses "IPMK501.XPOS" only instead of using "IPMK203.XPOS" and "IPMK203.YPOS".
Keep the file structure the same without making any removals. Just make the additions listed above.
'''