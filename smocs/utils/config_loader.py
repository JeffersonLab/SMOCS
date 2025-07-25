import yaml
import os
from typing import Dict, List, Any, Optional

class ConfigLoader:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
            
        return config or {}
    
    # MQTT Configuration Methods
    def has_mqtt_config(self) -> bool:
        return 'mqtt' in self.config and 'topics' in self.config['mqtt']
    
    def get_mqtt_topic_configs(self) -> List[Dict[str, Any]]:
        """Returns list of topic configurations with validation"""
        if not self.has_mqtt_config():
            return []
        
        topics = self.config['mqtt']['topics']
        validated_topics = []
        
        for topic_config in topics:
            # Validate required fields
            if 'topic' not in topic_config:
                raise ValueError("Each MQTT topic must have a 'topic' field")
            if 'channel_paths' not in topic_config:
                raise ValueError(f"Topic {topic_config['topic']} must have 'channel_paths'")
            
            validated_topics.append(topic_config)
        
        return validated_topics
    
    def get_mqtt_topics_list(self) -> List[str]:
        """Returns just the topic names for MQTT subscription"""
        return [config['topic'] for config in self.get_mqtt_topic_configs()]
    
    def get_topic_config(self, topic_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific topic"""
        for config in self.get_mqtt_topic_configs():
            if config['topic'] == topic_name:
                return config
        return None
    
    # Gymnasium Configuration Methods
    def has_gymnasium_config(self) -> bool:
        return 'gymnasium' in self.config
    
    def get_gymnasium_environment(self) -> str:
        return self.config.get('gymnasium', {}).get('environment', 'CartPole-v1')
    
    def get_gymnasium_input_topic(self) -> str:
        return self.config.get('gymnasium', {}).get('input_topic', 'gym-actions')
    
    def get_gymnasium_output_topic(self) -> str:
        return self.config.get('gymnasium', {}).get('output_topic', 'gym-observations')
    
    def get_gymnasium_blocking_mode(self) -> bool:
        return self.config.get('gymnasium', {}).get('blocking_mode', True)
    
    def get_gymnasium_default_action_strategy(self) -> str:
        return self.config.get('gymnasium', {}).get('default_action_strategy', 'random')
    
    def get_gymnasium_step_delay(self) -> float:
        return self.config.get('gymnasium', {}).get('step_delay', 0.0)
    
    def get_gymnasium_reset_on_start(self) -> bool:
        return self.config.get('gymnasium', {}).get('reset_on_start', True)
    
    def get_gymnasium_max_episode_steps(self) -> Optional[int]:
        return self.config.get('gymnasium', {}).get('max_episode_steps')
    
    def get_gymnasium_render_mode(self) -> Optional[str]:
        return self.config.get('gymnasium', {}).get('render_mode')
    
    def get_gymnasium_config(self) -> Dict[str, Any]:
        """Get complete gymnasium configuration with defaults"""
        return {
            'environment': self.get_gymnasium_environment(),
            'input_topic': self.get_gymnasium_input_topic(),
            'output_topic': self.get_gymnasium_output_topic(),
            'blocking_mode': self.get_gymnasium_blocking_mode(),
            'default_action_strategy': self.get_gymnasium_default_action_strategy(),
            'step_delay': self.get_gymnasium_step_delay(),
            'reset_on_start': self.get_gymnasium_reset_on_start(),
            'max_episode_steps': self.get_gymnasium_max_episode_steps(),
            'render_mode': self.get_gymnasium_render_mode()
        }
    
    # Kafka Configuration Methods
    def get_kafka_auto_create(self) -> bool:
        return self.config.get('kafka', {}).get('auto_create', True)
    
    def get_kafka_partitions(self) -> int:
        return self.config.get('kafka', {}).get('partitions', 1)
    
    def get_kafka_replication_factor(self) -> int:
        return self.config.get('kafka', {}).get('replication_factor', 1)
    
    # Project Configuration
    def get_project_name(self) -> str:
        return self.config.get('project', {}).get('name', 'smocs-project')