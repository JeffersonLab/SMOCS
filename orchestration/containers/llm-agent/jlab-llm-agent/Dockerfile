FROM llm-agent-base:latest

COPY ./containers/llm-agent/jlab-llm-agent/requirements.txt ./requirements-agent.txt
RUN pip install --no-cache-dir -r requirements-agent.txt

ENTRYPOINT ["chainlit", "run", "/app/orchestration/containers/llm-agent/chainlit_server.py", "--host", "0.0.0.0", "--port", "8000"]
