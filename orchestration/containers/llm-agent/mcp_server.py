import os
import yaml
import traceback
from fastmcp import FastMCP
import subprocess



mcp = FastMCP('my_mcp')



@mcp.tool()
def read_file(path: str) -> dict | list | str | int | float | bool | None:
    """
    Read a file from disk and return its contents.

    If the file extension is .yaml or .yml, the file is parsed as YAML and the
    deserialized Python object is returned. Otherwise, the raw file contents are
    returned as a string.

    Args:
        path: Path to an existing file.

    Returns:
        For .yaml/.yml files: the parsed content as dict, list, str, int, float,
        bool, or None depending on the YAML structure.
        For all other files: the full file contents as a string.
        If an error occurs, a string containing the full traceback is returned instead.
    """
    try:
        with open(path, 'r', encoding='utf-8') as file:
            if path.lower().endswith(('.yaml', '.yml')):
                return yaml.safe_load(file)
            else:
                return file.read()
    except Exception:
        return f'An error occurred during the execution of read_file:\n{traceback.format_exc()}'


@mcp.tool()
def write_file(path: str, val: dict | list | str | int | float | bool | None) -> str:
    """
    Write val to a file on disk.

    Mirrors read_file: if the extension is .yaml or .yml, val is serialized with
    yaml.safe_dump and written as YAML. Otherwise val is written as plain text
    (non-string values are serialized to YAML as a fallback).

    Args:
        path: Destination file path.
        val: For .yaml/.yml files: any JSON-serializable object. For all other files: a plain string.

    Returns:
        string indicating successful file writing or failure with the full traceback error message.

    Notes:
        - Existing files are overwritten.
        - Parent directories must already exist.
    """
    try:
        with open(path, 'w', encoding='utf-8') as file:
            if path.lower().endswith(('.yaml', '.yml')):
                yaml.safe_dump(val, file)
            else:
                if not isinstance(val, str):
                    val = yaml.safe_dump(val)
                file.write(val)
        return f'File "{path}" written successfully'
    except Exception:
        return f'An error occurred during the execution of write_file:\n{traceback.format_exc()}'


@mcp.tool()
def list_directory(path: str) -> str:
    """
    Recursively list the contents of a directory, including hidden files and directories.

    Args:
        path: Path to an existing directory.

    Returns:
        A string representing the directory tree, with each entry on its own line.
        Directories are marked with a trailing "/".
        If an error occurs, a string containing the full traceback is returned instead.

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
    except Exception:
        return f'An error occurred during the execution of list_directory:\n{traceback.format_exc()}'


@mcp.tool()
def launch_containers() -> str:
    """
    Rebuild and launch Docker Compose services for the project.

    This tool executes the following commands from the project
    compose directory:

        docker compose build --no-cache --pull
        docker compose up -d --force-recreate

    Returns:
        A status message indicating whether the containers were launched successfully, or the full traceback if an error occurred.

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
            'docker compose build --no-cache --pull && docker compose --env-file .env --env-file .env.secrets up -d --force-recreate',
            shell=True,
            cwd=compose_dir,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=None,
        )
        return 'Building images and launching containers in the background. Inform the user that the services are in the process of launching (which can take a while before they all become up and running) and that they can use the Docker CLI to check the status of the containers (e.g., `docker ps` to list running containers, `docker compose ps` to see service health, or `docker logs <container_name>` to inspect a specific container\'s output).'
    except Exception:
        return f'An error occurred during the execution of launch_containers:\n{traceback.format_exc()}'



if __name__ == '__main__':
    mcp.run(transport='stdio')      # 'stdio' is the default but adding for clarity
