# server.py
from fastmcp import FastMCP
import yaml
import os



mcp = FastMCP("yaml-reader")



@mcp.tool()
def read_yaml(path: str) -> dict:
    """
    Read a YAML file and return its contents as a JSON-serializable object.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
    return data



if __name__ == '__main__':
    mcp.run(transport='stdio')      # 'stdio' is the default but adding for clarity
