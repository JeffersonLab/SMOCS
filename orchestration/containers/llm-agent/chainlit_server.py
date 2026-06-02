import os
import re
import time
import yaml
import uuid
import myers
import inspect
from typing import Annotated, List, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages        # Instead of creating a wrapper tool node around the ToolNode to manually append the ToolMessage to state['messages']
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import ToolNode
import chainlit as cl


API_TYPE = os.environ.get('API_TYPE', '').lower()
if API_TYPE == 'openai':
    from langchain_openai import ChatOpenAI
elif API_TYPE == 'ollama':
    from langchain_ollama import ChatOllama
else:
    raise ValueError(f'Unrecognized API_TYPE: "{API_TYPE}". Must be one of: openai, ollama.')



class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    first_msg_idx_to_normalize: int = 0



async def get_agent(tools: list):
    """
    Build and return the LangGraph agent.
    Tools are passed in from the already-alive MCP client.
    """
    def normalize_new_messages_inplace(state) -> None:
        # Needed for OpenAI-compatible APIs: content can arrive as a list of dicts instead of a plain string.
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
            normalize_message_inplace(msgs[idx])
        state['first_msg_idx_to_normalize'] = len(msgs)


    def maybe_convert_to_tool_call(msg: AIMessage) -> AIMessage:
        # Needed for Ollama: some models output tool calls as JSON/YAML text instead of structured tool_calls.
        if getattr(msg, 'tool_calls', None):
            return msg
        text = msg.content
        match = re.search(
            r'```(?:json|yaml|yml)?\s*(.*?)\s*```',
            text,
            re.DOTALL | re.IGNORECASE
        )
        inner = match.group(1) if match else text.strip()
        try:
            data = yaml.safe_load(inner)
        except Exception:
            return msg
        if (not isinstance(data, dict)) or ('name' not in data) or ('arguments' not in data):
            return msg
        msg.tool_calls = [{
            'name': data['name'],
            'args': data['arguments'],
            'id': str(uuid.uuid4()),
            'type': 'tool_call'
        }]
        msg.content = ''
        print('Converted msg to LangGraph tool_call !')
        return msg


    async def llm_call(state: AgentState) -> dict:
        """Call LLM and return new message"""
        if API_TYPE == 'openai':
            normalize_new_messages_inplace(state)
        response = await llm.ainvoke(state['messages'])
        if API_TYPE == 'ollama':
            response = maybe_convert_to_tool_call(response)
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

        res = await cl.AskActionMessage(
            content=f'AI wants to use the following tool(s):\n\n{tool_text}\n\nDo you approve?',
            actions=[
                cl.Action(name='approve', label='✅ Approve', payload={'decision': 'yes'}),
                cl.Action(name='reject', label='❌ Reject', payload={'decision': 'no'}),
            ],
            timeout=600,  # 10 minutes timeout
        ).send()

        # Case 1: Approved — execute tools right away
        if res and res.get('payload', {}).get('decision') == 'yes':
            await cl.Message(content=f'✅ **Approved tool call(s):**\n\n{tool_text}').send()
            return {}

        # Case 4: Approve/Reject timeout — user did not respond
        if res is None:
            await cl.Message(content=f'⏰ **Tool call(s) timed out (not executed):**\n\n{tool_text}').send()
            stub_tool_msgs_timeout = [
                ToolMessage(content='Tool call approval timed out — tool was not executed.', tool_call_id=tc['id'])
                for tc in last_message.tool_calls
            ]
            return {
                'messages': stub_tool_msgs_timeout + [
                    SystemMessage(content=inspect.cleandoc('''
                        The tool approval prompt timed out — the user did not respond.
                        Let the user know the tools were not executed due to the timeout,
                        and ask if there is something they would still like to do.
                    '''))
                ]
            }

        # Case 2 & 3: Explicit reject — ask for feedback (timeout = user rejected but said nothing)
        await cl.Message(content=f'❌ **Explicitly rejected tool call(s):**\n\n{tool_text}').send()
        feedback_res = await cl.AskUserMessage(
            content='Tool execution blocked. Please provide feedback or a new instruction:',
            timeout=300,  # 5 minutes timeout
        ).send()

        stub_tool_msgs_rejected = [
            ToolMessage(content='Tool call rejected by user.', tool_call_id=tc['id'])
            for tc in last_message.tool_calls
        ]

        # Case 2: Explicit reject + feedback provided
        if feedback_res:
            return {'messages': stub_tool_msgs_rejected + [HumanMessage(content=feedback_res['output'])]}

        # Case 3: Explicit reject + feedback timeout — user rejected but gave no reason
        return {
            'messages': stub_tool_msgs_rejected + [
                SystemMessage(content=inspect.cleandoc('''
                    The user rejected the tool call(s) but did not provide a reason
                    (feedback timed out). Gently ask if they can explain why they
                    rejected, or if there is something else they need.
                '''))
            ]
        }


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
    if API_TYPE == 'openai':
        llm = ChatOpenAI(base_url=os.environ['LLM_URL'], api_key=os.environ['API_KEY'], model=os.environ['LLM_NAME'], temperature=0.0)
    elif API_TYPE == 'ollama':
        if os.environ.get('OLLAMA_NUM_CTX'):
            _num_ctx = int(os.environ['OLLAMA_NUM_CTX'])
        elif os.environ.get('OLLAMA_CONTEXT_LENGTH'):
            _num_ctx = int(os.environ['OLLAMA_CONTEXT_LENGTH'])
        else:
            _num_ctx = None
        llm = ChatOllama(model=os.environ['LLM_NAME'], temperature=0.0, num_ctx=_num_ctx)
    llm = llm.bind_tools(tools)

    # Build graph
    graph = StateGraph(AgentState)
    graph.add_node('llm_call', llm_call)
    graph.add_node('human_approval', human_approval)
    graph.add_node('tools', ToolNode(tools=tools, messages_key='messages', handle_tool_errors=True))
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


def get_opening_system_message() -> SystemMessage:
    docs_dir = os.path.join(os.path.dirname(__file__), '../../../SMOCS_DOCS')
    docs_sections = []
    for dirpath, dirnames, filenames in os.walk(docs_dir):
        dirnames.sort()
        for filename in sorted(filenames):
            if filename.endswith('.md'):
                filepath = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(filepath, docs_dir)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                docs_sections.append(f'### {rel_path}\n\n{content}')
    docs_content = '\n\n---\n\n'.join(docs_sections)

    system_message = SystemMessage(
        content = inspect.cleandoc(f'''
        You are a specialized assistant for the SMOCS system powered by {os.environ["LLM_NAME"]}.
        You help users read and answer questions about the current configuration, generate or edit configuration files, and launch the system.

        ## Rules
        - Do NOT invent fields, keys, or structures not present in the documentation below.
        - Do NOT make unrequested edits.
        - If the request is missing required information, ask a concise clarification question. Do NOT guess or assume missing keys/values.
        - Keep the data types the same when generating new configurations. For example, if a field is a list of strings in the documentation, it should NOT be changed to a single string or a dict.
        - When editing an existing configuration file, keep the structure and formatting as similar as possible to the original file. For example, if one of the fields in the original file is a list of ints and the user did not ask you to change it, do not unnecessarily change it to a list of floats.
        - Anytime you change one of the config files (i.e., 'config.yaml', 'docker-compose.yml', '.env'), you should check the others as well to keep them consistent. For example, if you add a new agent in `config.yaml`, you would also need to add that agent in `docker-compose.yml` to make sure it gets launched, and probably also add some relevant environment variables in `.env`. So please make sure to check all config files for consistency after any edit or generation.
        - Never edit the "jlab-llm-agent" or "ollama-agent" services in the docker-compose.yml file as those have nothing to do with the SMOCS system itself. They are just there to provide the user interface and LLM capabilities for you to assist the user with the SMOCS system.

        ## Available Resources
        - `/app/smocs/` — Python source code. Read this when the documentation below is insufficient.
        - `/app/orchestration/` — Live configuration files (`config.yaml`, `docker-compose.yml`, `.env`).

        ## SMOCS Documentation

        {docs_content}
        ''')
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
    cl.user_session.set('state', {'messages': [get_opening_system_message()], 'first_msg_idx_to_normalize': 1})      # first msg is already normalized. Its content is already a string.
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
        cl.user_session.set('state', {'messages': [get_opening_system_message()], 'first_msg_idx_to_normalize': 1})
        await stream_content('Chat history has been cleared. How may I assist you today?')
    else:
        agent = cl.user_session.get('agent')
        state = cl.user_session.get('state')
        state['messages'].append(HumanMessage(content=message.content))
        active_step = None
        active_step_t0 = None
        async for mode, chunk in agent.astream(state, stream_mode=['updates', 'debug']):
            if mode == 'debug':
                if chunk['type'] == 'task':
                    node_name = chunk['payload']['name']
                    if node_name == 'llm_call':
                        active_step = cl.Step(name='LLM...', type='llm')
                    elif node_name == 'tools':
                        active_step = cl.Step(name='Tool(s)...', type='tool')
                    # human_approval is skipped — AskActionMessage already provides its own UI feedback
                    if active_step:
                        await active_step.__aenter__()   # opens spinner in UI
                        active_step_t0 = time.perf_counter()
                elif chunk['type'] == 'task_result' and active_step:
                    elapsed = time.perf_counter() - active_step_t0
                    active_step.name = active_step.name.replace('...', f'  ({elapsed:.2f}s)')
                    await active_step.__aexit__(None, None, None)  # closes spinner, displays updated name
                    active_step = None
            elif mode == 'updates':
                # chunk is a dict with only 1 key-value pair, where the key is the name of the node that just returned an update, and the value is the state delta returned from that node (not the full state, just the new info to be merged into the state)
                # Only 1 key is present because the graph is designed such that there are no parallel edges, so only 1 node can return an update at a time.
                node_name = list(chunk.keys())[0]       # can be one of 'llm_call', 'human_approval', 'tools'
                state_delta = chunk[node_name]      # state_delta has what is returned from the nodes, not the aggregated full state
                if isinstance(state_delta, dict) and ('messages' in state_delta):
                    new_msgs = state_delta['messages']
                    state['messages'].extend(new_msgs)
                    for msg in new_msgs:
                        if isinstance(msg, AIMessage) and msg.content:
                            await stream_content(msg.content)
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



1) In config.yaml, don't change data types (ex: floats to ints) unnecessarily. Just add agent 3 configs.
2) In docker-compose.yml, you are making more changes than necessary. You only need to add the agent 3 service and its corresponding volume in the volumes section. You can keep the rest of the file exactly the same.
3) Any recommended edits for the .env file?

Overall, keep the changes in any file minimal and surgical to satisfy my requests.
'''