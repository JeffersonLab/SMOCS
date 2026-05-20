import os
import yaml
from fastmcp import FastMCP
import subprocess



mcp = FastMCP('my_mcp')



@mcp.tool()
def read_yaml(path: str) -> dict | list | str | int | float | bool | None:
    """
    Read and parse a YAML file from disk.

    Args:
        path: Path to an existing YAML file.

    Returns:
        The parsed YAML content as a JSON-serializable Python object.
        Depending on the YAML structure, the returned value may be:
        dict, list, str, int, float, bool, or None.
        If an error occurs, a string containing the error message is returned instead.

    Notes:
        Uses yaml.safe_load().
        This tool is intended for structured configuration files.
    """
    try:
        with open(path, 'r') as file:
            data = yaml.safe_load(file)
        return data
    except Exception as e:
        return f'An error occurred during the execution of read_yaml:\n{repr(e)}'


@mcp.tool()
def write_yaml(path: str, val: dict | list | str | int | float | bool | None) -> str:
    """
    Serialize and write structured data to a YAML file.

    Args:
        path: Destination file path.
        val: JSON-serializable Python object to write as YAML.

    Returns:
        string indicating successful file writing or failure with the traceback error message.

    Notes:
        - Existing files are overwritten.
        - Parent directories must already exist.
        - Uses yaml.safe_dump() with:
            - indent=4
            - sort_keys=True
            - block-style formatting
        - Intended for writing configuration or structured data files.
    """
    try:
        with open(path, 'w') as file:
            yaml.safe_dump(val, file, indent=4, sort_keys=True, default_flow_style=False)
        return f'File "{path}" written successfully'
    except Exception as e:
        return f'An error occurred during the execution of write_yaml:\n{repr(e)}'


@mcp.tool()
def read_file(path: str) -> str:
    """
    Read a text file from disk and return its contents.

    Args:
        path: Path to an existing text file.

    Returns:
        The full file contents as a string.
        If an error occurs, a string containing the error message is returned instead.

    Notes:
        Intended for reading regular text-based files such as:
        source code, logs, configuration files, or documentation.
    """
    try:
        with open(path, 'r', encoding='utf-8') as file:
            file_str = file.read()
        return file_str
    except Exception as e:
        return f'An error occurred during the execution of read_file:\n{repr(e)}'


@mcp.tool()
def write_file(path: str, val: str) -> str:
    """
    Write text val to a file on disk.

    Args:
        path: Destination file path.
        val: Text content to write.

    Returns:
        string indicating successful file writing or failure with the traceback error message.

    Notes:
        - Existing files are overwritten.
        - Parent directories must already exist.
        - File is written using UTF-8 encoding.
        - Intended for writing source code, configuration files, logs, or general text files.
    """
    try:
        with open(path, 'w', encoding='utf-8') as file:
            file.write(val)
        return f'File "{path}" written successfully'
    except Exception as e:
        return f'An error occurred during the execution of write_file:\n{repr(e)}'


@mcp.tool()
def list_directory(path: str) -> str:
    """
    Recursively list the contents of a directory, including hidden files and directories.

    Args:
        path: Path to an existing directory.

    Returns:
        A string representing the directory tree, with each entry on its own line.
        Directories are marked with a trailing "/".
        If an error occurs, a string containing the error message is returned instead.

    Notes:
        - Hidden files and directories (names starting with ".") are included.
        - Symbolic links are listed but not followed.
        - Output is sorted alphabetically at each level.
    """
    try:
        lines = []
        path = os.path.abspath(path)
        lines.append(f'{path}/')
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames.sort()
            filenames.sort()
            level = dirpath.replace(path, '').count(os.sep)
            indent = '    ' * level
            rel = os.path.relpath(dirpath, path)
            if rel != '.':
                lines.append(f'{indent}{os.path.basename(dirpath)}/')
            subindent = '    ' * (level + 1)
            for fname in filenames:
                lines.append(f'{subindent}{fname}')
        return '\n'.join(lines)
    except Exception as e:
        return f'An error occurred during the execution of list_directory:\n{repr(e)}'


@mcp.tool()
def launch_containers() -> str:
    """
    Rebuild and launch Docker Compose services for the project.

    This tool executes the following commands from the project
    compose directory:

        docker compose build --no-cache --pull
        docker compose up -d --force-recreate

    Returns:
        A status message indicating whether the containers were launched successfully or an error occurred.

    Notes:
        - Forces a full image rebuild without using Docker cache.
        - Recreates running containers even if configuration has not changed.
        - Runs services in detached mode.
        - Docker and Docker Compose must be installed and available in the execution environment.
        - The active Compose profiles and environment variables are determined by the project's docker-compose.yml and .env configuration.
    """
    try:
        compose_dir = os.path.join(os.path.dirname(__file__), '../..')
        subprocess.Popen(
            'docker compose build --no-cache --pull && docker compose up -d --force-recreate',
            shell=True,
            cwd=compose_dir,
            stdout=subprocess.DEVNULL,
            stderr=None,
        )
        return 'Building images and launching containers in the background ... It might take a few moments for all services to be up and running.'
    except Exception as e:
        return f'An error occurred during the execution of launch_containers:\n{repr(e)}'



if __name__ == '__main__':
    mcp.run(transport='stdio')      # 'stdio' is the default but adding for clarity