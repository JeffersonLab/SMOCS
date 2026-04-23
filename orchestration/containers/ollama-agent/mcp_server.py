import os
import yaml
from fastmcp import FastMCP
import subprocess
import requests
import time



mcp = FastMCP('my_mcp')



@mcp.tool()
def read_yaml(path: str) -> dict | list | str | int | float | bool | None:
    """
    Read a YAML file and return its contents as a JSON-serializable object.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
    return data


@mcp.tool()
def read_file(path: str) -> str:
    """
    Read a file and return its contents as a string.
    """
    with open(path, 'r') as file:
        file_str = file.read()
    return file_str


@mcp.tool()
def write_yaml(path: str, val: dict | list | str | int | float | bool | None) -> None:
    """
    Writes val to a YAML file specified in path.
    """
    with open(path, 'w') as file:
        yaml.safe_dump(val, file, indent=4, sort_keys=True, default_flow_style=False)


# @mcp.tool()
# def launch_services(services: list[str] = [], compose_filepath: str = './orchestration/docker-compose.yml') -> None:
#     """
#     Launches all services specified in the "services" list which are all written in compose_filepath. If "services" is empty, all services in the compose_filepath file will be launched.
#     """
#     command = ['docker', 'compose', 'up', '-d', '--build'] + services
#     compose_dir = os.path.dirname(compose_filepath)
#     subprocess.run(command, check=True, cwd=compose_dir)


@mcp.tool()
def launch_containers(release_ollama_gpu: bool = False) -> str:
    """
    Launches containers defined in the docker-compose.yml file based on the profiles defined in the .env file.
    Set release_ollama_gpu=True only if using an ollama model that is hosted on a local GPU that the containers also need.
    """
    compose_dir = os.path.join(os.path.dirname(__file__), '../..')
    llm_name = os.environ['LLM_NAME']
    ollama_url = 'http://localhost:11434/api/generate'

    # Step 1: Optionally release GPU
    if release_ollama_gpu:
        try:
            requests.post(ollama_url, json={'model': llm_name, 'keep_alive': 0})
        except Exception as e:
            return f"Failed to release GPU: {e}"

    # Step 2: Launch containers
    try:
        subprocess.run(
            ['docker', 'compose', 'build', '--no-cache'],
            check=True,
            cwd=compose_dir,
            stdout=subprocess.DEVNULL,
            stderr=None,
        )
        subprocess.run(
            ['docker', 'compose', 'up', '-d', '--force-recreate'],
            check=True,
            cwd=compose_dir,
            stdout=subprocess.DEVNULL,
            stderr=None,
        )
    except subprocess.CalledProcessError as e:
        if release_ollama_gpu:
            requests.post(ollama_url, json={'model': llm_name, 'keep_alive': -1, 'prompt': ''})
        return f"Failed to launch containers: {e}"

    # Step 3: Wait for all containers to be running/healthy
    timeout, interval, elapsed, statuses = 300, 5, 0, {}
    while elapsed < timeout:
        time.sleep(interval)
        elapsed += interval
        try:
            result = subprocess.run(
                ['docker', 'compose', 'ps', '--format', 'json'],
                check=True, cwd=compose_dir, capture_output=True, text=True,
            )
            containers = [json.loads(line) for line in result.stdout.strip().splitlines() if line]
            statuses = {container['Name']: container.get('Health', '') for container in containers}
            not_ready = [
                container['Name'] for container in containers
                if (container.get('State') != 'running') or (container.get('Health') in {'starting', 'unhealthy'})
            ]
            if len(not_ready) == 0:
                break
        except Exception as e:
            if release_ollama_gpu:
                requests.post(ollama_url, json={'model': llm_name, 'keep_alive': -1, 'prompt': ''})
            return f"Containers launched but failed to verify health: {e}"
    else:
        if release_ollama_gpu:
            requests.post(ollama_url, json={'model': llm_name, 'keep_alive': -1, 'prompt': ''})
        return f"Timed out waiting for containers after {timeout}s. Last statuses: {statuses}"

    return f"All containers launched successfully: {list(statuses.keys())}"



# TODO: mcp tool to get the tree directory structure of the smocs package



if __name__ == '__main__':
    mcp.run(transport='stdio')      # 'stdio' is the default but adding for clarity