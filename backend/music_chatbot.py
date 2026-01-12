#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MusicNote Chatbot - Interactive Music AI Assistant for MusicLab
Provides expert guidance on music theory, pattern selection, and model usage

Created on Dec 17 2025
@author: andreyvlasenko
"""

import yaml
from groq import Groq
from typing import List, Dict
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class MusicChatbot:
    """
    AI Music Assistant that helps users understand MusicLab
    and make better creative decisions
    """
    
    def __init__(self, config_path: str = "config/config_key.yaml"):
        """
        Initialize chatbot with configuration
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.groq_client = Groq(api_key=self.config['groq']['api_key'])
        self.model = self.config['groq']['model']
        self.temperature = self.config['groq']['temperature']
        self.max_tokens = self.config['groq']['max_tokens']
        self.system_prompt = self.config['musiclab']['system_prompt']
        self.max_history = self.config['musiclab']['max_history']
        
        # Session-based conversation histories (in-memory)
        self.sessions = {}
        
        logger.info("MusicChatbot initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def _get_session_history(self, session_id: str) -> List[Dict]:
        """Get or create conversation history for a session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        return self.sessions[session_id]
    
    def _add_to_history(self, session_id: str, role: str, content: str):
        """Add message to session history with automatic trimming"""
        history = self._get_session_history(session_id)
        history.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last max_history exchanges (2 messages per exchange)
        max_messages = self.max_history * 2
        if len(history) > max_messages:
            self.sessions[session_id] = history[-max_messages:]
    
    def chat(self, session_id: str, user_message: str) -> Dict:
        """
        Process user message and return AI response
        
        Args:
            session_id: Unique session identifier
            user_message: User's question or message
            
        Returns:
            Dict with response, timestamp, and status
        """
        try:
            # Add user message to history
            self._add_to_history(session_id, 'user', user_message)
            
            # Build messages for Groq API
            messages = [
                {'role': 'system', 'content': self.system_prompt}
            ]
            
            # Add conversation history (without timestamps)
            history = self._get_session_history(session_id)
            for msg in history:
                messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
            
            logger.info(f"Session {session_id}: Sending {len(messages)} messages to Groq")
            
            # Call Groq API
            chat_completion = self.groq_client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=1,
                stream=False
            )
            
            # Extract response
            assistant_message = chat_completion.choices[0].message.content
            
            # Add assistant response to history
            self._add_to_history(session_id, 'assistant', assistant_message)
            
            logger.info(f"Session {session_id}: Response generated successfully")
            
            return {
                'status': 'success',
                'response': assistant_message,
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id,
                'history_length': len(self._get_session_history(session_id)) // 2
            }
            
        except Exception as e:
            logger.error(f"Chat error for session {session_id}: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'response': "Sorry, I encountered an error. Please try again! 🎵",
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id
            }
    
    def clear_session(self, session_id: str):
        """Clear conversation history for a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Cleared session {session_id}")
    
    def get_session_info(self, session_id: str) -> Dict:
        """Get information about a session"""
        history = self._get_session_history(session_id)
        return {
            'session_id': session_id,
            'message_count': len(history),
            'exchanges': len(history) // 2,
            'exists': session_id in self.sessions
        }


# Singleton instance for the application
_chatbot_instance = None


def get_chatbot_instance(config_path: str = "config/config_key.yaml") -> MusicChatbot:
    """Get or create singleton chatbot instance"""
    global _chatbot_instance
    if _chatbot_instance is None:
        _chatbot_instance = MusicChatbot(config_path)
    return _chatbot_instance
