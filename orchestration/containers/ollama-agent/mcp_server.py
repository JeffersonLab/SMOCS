import os
import yaml
from fastmcp import FastMCP



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


# TODO: mcp tool to get the tree directory structure of the smocs package



if __name__ == '__main__':
    mcp.run(transport='stdio')      # 'stdio' is the default but adding for clarity