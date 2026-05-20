import os
import re
import time
import yaml
import uuid
import myers
from typing import Annotated, List, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages        # Instead of creating a wrapper tool node around the ToolNode to manually append the ToolMessage to state['messages']
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import ToolNode
import chainlit as cl



class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    first_msg_idx_to_normalize: int = 0



async def get_agent(tools: list):
    """
    Build and return the LangGraph agent.
    Tools are passed in from the already-alive MCP client.
    """
    def normalize_new_messages_inplace(state) -> None:
        def normalize_message_inplace(msg: BaseMessage) -> None:
            if isinstance(msg.content, str):
                pass
            elif isinstance(msg.content, list):
                msg_content = []
                for part in msg.content:
                    if isinstance(part, dict):
                        msg_content.append(part.get('text', str(part)))
                    elif isinstance(part, str):
                        msg_content.append(part)
                    else:
                        raise TypeError(f'Expecting each part in message to be either a dict or str, but found "{type(part)}" !!!')
                msg.content = '\n'.join(msg_content)
            else:
                raise TypeError(f'Expecting the content of each message to be either a str or list, but found "{type(msg.content)}" !!!')
            return None          
        
        msgs = state['messages']
        first_msg_idx_to_normalize = state.get('first_msg_idx_to_normalize', 0)
        for idx in range(first_msg_idx_to_normalize, len(msgs), 1):
            #print(f'{type(msgs[idx])} (id: {msgs[idx].id}) BEFORE normalization - content: {msgs[idx].content}')
            normalize_message_inplace(msgs[idx])
            #print(f'{type(msgs[idx])} (id: {msgs[idx].id}) AFTER normalization - content: {msgs[idx].content}')
        state['first_msg_idx_to_normalize'] = len(msgs)
    

    async def llm_call(state: AgentState) -> dict:
        """Call LLM and return new message"""
        normalize_new_messages_inplace(state)
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
            if tool_call['name'] in {'write_yaml', 'write_file'}:
                if 'path' not in tool_call['args']:
                    return {
                        'messages': [ToolMessage(content='Error: Did not find "path" in tool_call["args"] !', tool_call_id=tool_call['id'])]
                    }
                if 'val' not in tool_call['args']:
                    return {
                        'messages': [ToolMessage(content='Error: Did not find "val" in tool_call["args"] !', tool_call_id=tool_call['id'])]
                    }

                filepath = tool_call['args']['path']
                val = tool_call['args']['val']
                # write_yaml: val is a dict, so serialize to YAML string for diffing.
                # write_file: val is already a plain-text string.
                if tool_call['name'] == 'write_yaml':
                    val_str = yaml.safe_dump(val, sort_keys=True, indent=4, default_flow_style=False)
                else:
                    val_str = val
                if os.path.isfile(filepath):
                    # Apply myers algorithm to determine the shortest path of changes from old_val_str to val_str
                    # NOTE: difflib.ndiff did NOT work well when there are common sections. It shows the changes in unexpected places.
                    with open(filepath, 'r', encoding='utf-8') as file:
                        # write_yaml: round-trip through yaml to normalize formatting before diffing.
                        # write_file: read as plain text.
                        if tool_call['name'] == 'write_yaml':
                            old_val_str = yaml.safe_dump(yaml.safe_load(file), sort_keys=True, indent=4, default_flow_style=False)
                        else:
                            old_val_str = file.read()
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
                    txt = f"### Tool {i}: `{tool_call['name']}`\n**📝 Overwriting File:** `{filepath}`\n```diff\n{final_output}\n```"
                else:
                    txt = f"### Tool {i}: `{tool_call['name']}`\n**📝 File:** `{filepath}`\n```diff\n{val_str}\n```"
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
    llm = ChatOpenAI(base_url=os.environ['LLM_URL'], api_key=os.environ['LLM_KEY'], model=os.environ['LLM_NAME'], temperature=0.0)
    llm = llm.bind_tools(tools)

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


def get_system_message() -> SystemMessage:
    system_message = SystemMessage(
        content = '''
        You are a specialized assistant for the SMOCS system.
        You help users read and answer questions about the current configuration, generate or edit configuration files, and launch the system.

        ## Rules
        - Do NOT invent fields, keys, or structures not present in the documentation.
        - Do NOT make unrequested edits.
        - If the request is missing required information, ask a concise clarification question. Do NOT guess or assume missing values.

        ## Available Resources
        - `/app/SMOCS_DOCS/` — Project documentation (Markdown). This is the primary source of truth for allowed configuration structures and valid field values.
        - `/app/smocs/` — Python source code. Read this when documentation alone is insufficient.
        - `/app/orchestration/` — Live configuration files (`config.yaml`, `docker-compose.yml`, `.env`).

        At the beginning of the chat, start by reading the documentation to understand the mechanics of the SMOCS system.
        Refer to the code if the user's request is not fully addressable by the documentation alone.
        '''.strip()
    )
    return system_message


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
            'env': os.environ.copy(),   # forward all parent env vars to the mcp subprocess
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
    cl.user_session.set('state', {'messages': [get_system_message()], 'first_msg_idx_to_normalize': 1})      # first msg is already normalized. Its content is already a string.
    #cl.user_session.set('state', {'messages': [], 'first_msg_idx_to_normalize': 0})
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
        cl.user_session.set('state', {'messages': [get_system_message()], 'first_msg_idx_to_normalize': 1})
        #cl.user_session.set('state', {'messages': [], 'first_msg_idx_to_normalize': 0})
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
Show me all the configs of agent 2 then.
d)
Add the following to the configurations:
1. Append a new PV called "IPMK501.XPOS" to the list of epics PVs
2. Add a new autoencoder called "autoencoder_agent3" that is identical to "autoencoder_agent2". The only difference is that it uses "IPMK501.XPOS" only instead of using "IPMK203.XPOS" and "IPMK203.YPOS".
Keep the configuration structure the same without making any removals. Just make the additions listed above. Then save those configs to the same path (i.e., overwrite the file).
e)
Launch containers
'''