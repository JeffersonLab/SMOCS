import os
import yaml
from fastmcp import FastMCP




mcp = FastMCP('my_mcp')



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


@mcp.tool()
def read_file(path: str) -> str:
    """
    Read a file and return its contents as a string.
    """
    with open(path, 'r') as file:
        file_str = file.read()
    return file_str


# TODO: mcp tool to get the tree directory structure of the smocs package



if __name__ == '__main__':
    mcp.run(transport='stdio')      # 'stdio' is the default but adding for clarity