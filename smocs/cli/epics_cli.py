#!/usr/bin/env python3
"""
CLI client for EPICS producer control.
Single command mode - connects, sends command, prints response, exits.
"""

import socket
import json
import sys


SOCKET_PATH = '/tmp/epics-producer.sock'


def send_command(command_dict):
    """
    Send command to producer and return response.
    
    Args:
        command_dict: Dictionary containing command and parameters
        
    Returns:
        Response dictionary from producer
    """
    try:
        # Connect to Unix socket
        client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client_socket.connect(SOCKET_PATH)
        
        # Send command
        command_json = json.dumps(command_dict)
        client_socket.sendall(command_json.encode('utf-8'))
        
        # Receive response
        response_data = client_socket.recv(4096).decode('utf-8')
        response = json.loads(response_data)
        
        client_socket.close()
        
        return response
        
    except FileNotFoundError:
        return {
            'status': 'error',
            'message': 'Error: Cannot connect to producer'
        }
    except ConnectionRefusedError:
        return {
            'status': 'error',
            'message': 'Error: Cannot connect to producer'
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Error: {str(e)}'
        }


def print_help():
    """Print usage information."""
    print("Usage: epics-cli <command> [arguments]")
    print()
    print("Commands:")
    print("  add-pv <pv_name>       Add a PV to monitor")
    print("  remove-pv <pv_name>    Remove a PV from monitoring")
    print("  list-pvs               List all monitored PVs")
    print("  set-source <name>      Change Kafka topic name")
    print("  status                 Show current status")
    print("  help                   Show this help message")
    print()
    print("Examples:")
    print("  epics-cli add-pv IBCAD00CRCUR7")
    print("  epics-cli list-pvs")
    print("  epics-cli set-source JLAB-BACKUP")


def format_response(response):
    """Format and print response from producer."""
    status = response.get('status', 'unknown')
    message = response.get('message', '')
    
    if status == 'error':
        print(f"Error: {message}")
        return
    
    # Print message if present
    if message:
        print(message)
    
    # Print PVs and source if present
    pvs = response.get('pvs')
    source = response.get('source')
    
    if pvs is not None or source is not None:
        if pvs:
            print(f"PVs: {', '.join(pvs)}")
        else:
            print("PVs: (none)")
        
        if source:
            print(f"Source: {source}")


def main():
    """Main entry point for CLI."""
    # Parse command line arguments
    if len(sys.argv) < 2:
        print_help()
        sys.exit(0)
    
    command = sys.argv[1]
    
    # Handle help command
    if command == 'help':
        print_help()
        sys.exit(0)
    
    # Build command dictionary
    command_dict = {'command': command}
    
    # Parse command-specific arguments
    if command == 'add-pv' or command == 'add_pv':
        command_dict['command'] = 'add_pv'
        if len(sys.argv) < 3:
            print("Error: PV name required")
            print("Usage: epics-cli add-pv <pv_name>")
            sys.exit(1)
        command_dict['pv'] = sys.argv[2]
    
    elif command == 'remove-pv' or command == 'remove_pv':
        command_dict['command'] = 'remove_pv'
        if len(sys.argv) < 3:
            print("Error: PV name required")
            print("Usage: epics-cli remove-pv <pv_name>")
            sys.exit(1)
        command_dict['pv'] = sys.argv[2]
    
    elif command == 'list-pvs' or command == 'list_pvs':
        command_dict['command'] = 'list_pvs'
    
    elif command == 'set-source' or command == 'set_source':
        command_dict['command'] = 'set_source'
        if len(sys.argv) < 3:
            print("Error: Source name required")
            print("Usage: epics-cli set-source <name>")
            sys.exit(1)
        command_dict['source'] = sys.argv[2]
    
    elif command == 'status':
        command_dict['command'] = 'status'
    
    else:
        print(f"Error: Unknown command '{command}'")
        print("Use 'epics-cli help' to see available commands")
        sys.exit(1)
    
    # Send command and print response
    response = send_command(command_dict)
    format_response(response)


if __name__ == '__main__':
    main()