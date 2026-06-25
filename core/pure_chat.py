"""
Pure AI Chat System
No Ollama, no pre-trained models
Just pure learning from scratch
"""

import os
import json
from datetime import datetime
from core.pure_learning import PureLearningSystem
from core.chat_history import get_global_history, add_global_message

# Initialize the pure learning system
pure_ai = PureLearningSystem()

def chat(user_input, username):
    """
    Chat with the pure AI
    The AI learns and responds based on what it has learned
    """
    # Record interaction
    pure_ai.interactions.append({
        "input": user_input,
        "username": username,
        "timestamp": datetime.now().isoformat()
    })
    
    # Learn from input
    input_words = user_input.lower().split()
    
    # Learn each word
    for word in input_words:
        pure_ai.learn_word(word)
    
    # Learn meaning from context
    if len(input_words) >= 2:
        for word in input_words:
            pure_ai.learn_meaning(word, user_input)
    
    # Learn patterns
    pure_ai.learn_pattern(input_words)
    
    # Generate response
    response = pure_ai.generate_response(user_input)
    
    # Record conversation
    add_global_message("user", f"[{username}] {user_input}")
    add_global_message("assistant", response)
    
    return response

def get_ai_state():
    """Get current state of the AI"""
    stats = pure_ai.get_stats()
    return {
        "vocabulary_size": stats["vocabulary_size"],
        "patterns_learned": stats["patterns_learned"],
        "concepts_learned": stats["concepts_learned"],
        "total_interactions": stats["total_interactions"],
        "knowledge_level": stats["knowledge_level"]
    }

def get_ai_stats():
    """Get detailed statistics"""
    return pure_ai.get_stats()
