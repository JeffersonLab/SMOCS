import socket
import json
import threading
import logging
import os


class EpicsCLIController:
    """
    CLI controller for EPICS producer using Unix socket communication.
    Handles command registration, socket server, and command routing.
    """
    
    def __init__(self, target_object):
        """
        Initialize the CLI controller.
        
        Args:
            target_object: The object (EpicsKafkaProducer) to control
        """
        self.target = target_object
        self.commands = {}
        self.running = False
        self.thread = None
        self.socket_path = '/tmp/epics-producer.sock'
        self.server_socket = None
        
        logging.info("EpicsCLIController initialized")
    
    def register_command(self, name, handler, help_text):
        """
        Register a CLI command.
        
        Args:
            name: Command name (e.g., 'add_pv')
            handler: Method to call when command is received
            help_text: Description of the command
        """
        self.commands[name] = {
            'handler': handler,
            'help': help_text
        }
        logging.debug(f"Registered CLI command: {name}")
    
    def start(self):
        """
        Start the CLI control server in a background thread.
        
        Raises:
            Exception: If socket setup fails (fatal for producer)
        """
        try:
            # Remove old socket file if it exists
            if os.path.exists(self.socket_path):
                os.remove(self.socket_path)
            
            # Create Unix socket
            self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.server_socket.bind(self.socket_path)
            self.server_socket.listen(1)
            
            # Set socket permissions
            os.chmod(self.socket_path, 0o666)
            
            logging.info(f"CLI control socket listening on {self.socket_path}")
            
            # Start background thread
            self.running = True
            self.thread = threading.Thread(target=self._socket_server_loop, daemon=True)
            self.thread.start()
            
            logging.info("CLI control server started")
            
        except Exception as e:
            logging.error(f"Failed to start CLI control server: {e}")
            raise  # Fatal - producer should not start without CLI if enabled
    
    def stop(self):
        """Stop the CLI control server and clean up resources."""
        logging.info("Stopping CLI control server...")
        self.running = False
        
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception as e:
                logging.error(f"Error closing server socket: {e}")
        
        if os.path.exists(self.socket_path):
            try:
                os.remove(self.socket_path)
            except Exception as e:
                logging.error(f"Error removing socket file: {e}")
        
        logging.info("CLI control server stopped")
    
    def _socket_server_loop(self):
        """Main loop for accepting socket connections and processing commands."""
        while self.running:
            try:
                # Accept one connection at a time
                client_socket, _ = self.server_socket.accept()
                
                try:
                    # Receive command (max 4096 bytes)
                    data = client_socket.recv(4096).decode('utf-8')
                    
                    if not data:
                        continue
                    
                    # Parse and handle command
                    try:
                        command_dict = json.loads(data)
                        response = self._handle_command(command_dict)
                    except json.JSONDecodeError as e:
                        response = {
                            'status': 'error',
                            'message': 'Invalid JSON format'
                        }
                    
                    # Send response
                    response_json = json.dumps(response)
                    client_socket.sendall(response_json.encode('utf-8'))
                    
                except Exception as e:
                    logging.error(f"Error handling client connection: {e}")
                    error_response = {
                        'status': 'error',
                        'message': str(e)
                    }
                    try:
                        client_socket.sendall(json.dumps(error_response).encode('utf-8'))
                    except:
                        pass
                
                finally:
                    client_socket.close()
                    
            except Exception as e:
                if self.running:
                    logging.error(f"Error in socket server loop: {e}")
    
    def _handle_command(self, command_dict):
        """
        Route command to appropriate handler.
        
        Args:
            command_dict: Dictionary with 'command' key and parameters
            
        Returns:
            Response dictionary with status and data
        """
        command = command_dict.get('command')
        
        if not command:
            return {
                'status': 'error',
                'message': 'No command specified'
            }
        
        if command == 'help':
            return self._generate_help()
        
        if command not in self.commands:
            return {
                'status': 'error',
                'message': f'Unknown command: {command}',
                'help': 'Use "help" command to see available commands'
            }
        
        # Call registered handler
        try:
            handler = self.commands[command]['handler']
            return handler(command_dict)
        except Exception as e:
            logging.error(f"Error executing command '{command}': {e}")
            return {
                'status': 'error',
                'message': f'Command failed: {str(e)}'
            }
    
    def _generate_help(self):
        """Generate help text showing available commands."""
        help_lines = ['Available commands:']
        for cmd_name, cmd_info in self.commands.items():
            help_lines.append(f"  {cmd_name}: {cmd_info['help']}")
        
        return {
            'status': 'ok',
            'message': '\n'.join(help_lines)
        }