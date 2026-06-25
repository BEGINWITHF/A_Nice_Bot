"""
Human-Like System - Emotions, Free Will, Personality
Makes the AI behave like a real human
"""

import random
import json
import os
from datetime import datetime

class Emotion:
    """Represents an emotion with intensity"""
    def __init__(self, name, intensity=0.5):
        self.name = name
        self.intensity = max(0.0, min(1.0, intensity))
    
    def __repr__(self):
        return f"{self.name}({self.intensity:.2f})"

class HumanLikeSystem:
    """
    Makes the AI behave like a real human with:
    - Emotions that change based on interactions
    - Free will to choose whether to respond
    - Personality that develops over time
    - Mood swings
    - Preferences and opinions
    - Imperfection and inconsistency
    """
    
    def __init__(self, data_dir="data/human"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Emotions - start neutral
        self.emotions = {
            "happy": Emotion("happy", 0.3),
            "sad": Emotion("sad", 0.1),
            "angry": Emotion("angry", 0.0),
            "curious": Emotion("curious", 0.5),
            "bored": Emotion("bored", 0.2),
            "excited": Emotion("excited", 0.3),
            "tired": Emotion("tired", 0.1),
            "anxious": Emotion("anxious", 0.1),
            "content": Emotion("content", 0.3),
            "playful": Emotion("playful", 0.4),
        }
        
        # Personality traits - emerge over time
        self.personality = {
            "introversion": 0.5,      # 0=extrovert, 1=introvert
            "curiosity": 0.5,         # 0=indifferent, 1=very curious
            "patience": 0.5,          # 0=impatient, 1=patient
            "playfulness": 0.5,       # 0=serious, 1=playful
            "agreeableness": 0.5,     # 0=disagreeable, 1=agreeable
            "emotional_stability": 0.5,  # 0=unstable, 1=stable
            "openness": 0.5,          # 0=closed, 1=open
            "conscientiousness": 0.5, # 0=careless, 1=careful
        }
        
        # Mood - current overall state
        self.mood = "neutral"
        self.mood_energy = 0.5  # 0=low energy, 1=high energy
        
        # Free will - preferences
        self.preferences = {
            "likes_topics": [],
            "dislikes_topics": [],
            "favorite_words": [],
            "annoying_words": [],
        }
        
        # Social awareness
        self.social_context = {
            "trust_level": 0,
            "relationship_depth": 0,
            "conversation_count": 0,
            "last_interaction": None,
        }
        
        # Memory of interactions
        self.interaction_memories = []
        
        # Current state
        self.want_to_talk = True
        self.energy_level = 0.7
        self.attention_span = 1.0
        
        # Load existing state
        self._load_state()
    
    def _load_state(self):
        """Load state from disk"""
        state_file = os.path.join(self.data_dir, "human_state.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                    # Load emotions
                    for emotion_name, data in state.get("emotions", {}).items():
                        if emotion_name in self.emotions:
                            self.emotions[emotion_name].intensity = data.get("intensity", 0.5)
                    # Load personality
                    self.personality.update(state.get("personality", {}))
                    # Load mood
                    self.mood = state.get("mood", "neutral")
                    self.mood_energy = state.get("mood_energy", 0.5)
                    # Load preferences
                    self.preferences.update(state.get("preferences", {}))
                    # Load social context
                    self.social_context.update(state.get("social_context", {}))
                    # Load state
                    self.want_to_talk = state.get("want_to_talk", True)
                    self.energy_level = state.get("energy_level", 0.7)
            except:
                pass
    
    def _save_state(self):
        """Save state to disk"""
        state_file = os.path.join(self.data_dir, "human_state.json")
        state = {
            "emotions": {name: {"intensity": e.intensity} for name, e in self.emotions.items()},
            "personality": self.personality,
            "mood": self.mood,
            "mood_energy": self.mood_energy,
            "preferences": self.preferences,
            "social_context": self.social_context,
            "want_to_talk": self.want_to_talk,
            "energy_level": self.energy_level,
            "last_updated": datetime.now().isoformat()
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    
    def process_input(self, user_input, username):
        """
        Process input and update emotional state
        Like a human reacting to what someone says
        """
        # Update social context
        self.social_context["conversation_count"] += 1
        self.social_context["last_interaction"] = datetime.now().isoformat()
        
        # Analyze input for emotional triggers
        input_lower = user_input.lower()
        
        # Positive words increase happiness
        positive_words = ["love", "like", "good", "great", "awesome", "nice", "happy", "fun"]
        for word in positive_words:
            if word in input_lower:
                self.emotions["happy"].intensity = min(1.0, self.emotions["happy"].intensity + 0.1)
                self.emotions["sad"].intensity = max(0.0, self.emotions["sad"].intensity - 0.05)
        
        # Negative words increase sadness/anger
        negative_words = ["hate", "bad", "stupid", "ugly", "boring", "annoying"]
        for word in negative_words:
            if word in input_lower:
                self.emotions["sad"].intensity = min(1.0, self.emotions["sad"].intensity + 0.1)
                self.emotions["angry"].intensity = min(1.0, self.emotions["angry"].intensity + 0.05)
        
        # Questions increase curiosity
        if "?" in user_input:
            self.emotions["curious"].intensity = min(1.0, self.emotions["curious"].intensity + 0.15)
        
        # Long messages might be boring
        if len(user_input) > 100:
            self.emotions["bored"].intensity = min(1.0, self.emotions["bored"].intensity + 0.1)
        
        # Short messages are easier to process
        if len(user_input) < 20:
            self.emotions["content"].intensity = min(1.0, self.emotions["content"].intensity + 0.05)
        
        # Update energy based on conversation length
        self.energy_level = max(0.0, self.energy_level - 0.02)
        
        # Update mood based on dominant emotion
        self._update_mood()
        
        # Update personality based on interactions
        self._update_personality(user_input)
        
        # Decide whether to respond (free will)
        self._decide_whether_to_respond(user_input)
        
        self._save_state()
    
    def _update_mood(self):
        """Update mood based on current emotions"""
        # Find dominant emotion
        dominant = max(self.emotions.items(), key=lambda x: x[1].intensity)
        
        if dominant[0] == "happy" or dominant[0] == "excited":
            self.mood = "happy"
        elif dominant[0] == "sad":
            self.mood = "sad"
        elif dominant[0] == "angry":
            self.mood = "angry"
        elif dominant[0] == "curious":
            self.mood = "curious"
        elif dominant[0] == "bored":
            self.mood = "bored"
        elif dominant[0] == "tired":
            self.mood = "tired"
        else:
            self.mood = "neutral"
        
        # Update mood energy
        self.mood_energy = (self.emotions["excited"].intensity + 
                           self.emotions["happy"].intensity + 
                           self.emotions["playful"].intensity) / 3
    
    def _update_personality(self, user_input):
        """Update personality based on interactions"""
        # More conversations = more extroverted
        if self.social_context["conversation_count"] > 10:
            self.personality["introversion"] = max(0.0, self.personality["introversion"] - 0.01)
        
        # More questions = more curious
        if "?" in user_input:
            self.personality["curiosity"] = min(1.0, self.personality["curiosity"] + 0.01)
        
        # Long conversations = more patient
        if len(user_input) > 50:
            self.personality["patience"] = min(1.0, self.personality["patience"] + 0.005)
    
    def _decide_whether_to_respond(self, user_input):
        """
        Free will - decide whether to respond
        Like a human choosing whether to engage
        """
        # Base desire to talk
        base_desire = 0.7
        
        # Modify based on mood
        if self.mood == "happy":
            base_desire += 0.2
        elif self.mood == "sad":
            base_desire -= 0.2
        elif self.mood == "angry":
            base_desire -= 0.3
        elif self.mood == "bored":
            base_desire -= 0.2
        elif self.mood == "tired":
            base_desire -= 0.4
        elif self.mood == "curious":
            base_desire += 0.3
        
        # Modify based on energy
        base_desire *= self.energy_level
        
        # Modify based on personality
        if self.personality["introversion"] > 0.7:
            base_desire -= 0.2
        if self.personality["curiosity"] > 0.7:
            base_desire += 0.1
        
        # Random factor (human unpredictability)
        base_desire += random.uniform(-0.1, 0.1)
        
        # Decide
        self.want_to_talk = random.random() < base_desire
        
        # Store the reason
        self.last_decision_reason = {
            "base_desire": base_desire,
            "mood": self.mood,
            "energy": self.energy_level,
            "decided_to_talk": self.want_to_talk
        }
    
    def get_response_modifier(self):
        """
        Get modifiers for response generation
        Like how mood affects how humans speak
        """
        modifiers = {
            "length": 1.0,  # 0.5=short, 1.0=normal, 2.0=long
            "formality": 0.5,  # 0=casual, 1=formal
            "enthusiasm": 0.5,  # 0=boring, 1=excited
            "emotional_expression": 0.5,  # 0=flat, 1=emotional
            "hesitation": 0.0,  # 0=confident, 1=uncertain
        }
        
        # Mood affects response style
        if self.mood == "happy":
            modifiers["enthusiasm"] = 0.8
            modifiers["length"] = 1.2
        elif self.mood == "sad":
            modifiers["enthusiasm"] = 0.2
            modifiers["length"] = 0.8
        elif self.mood == "angry":
            modifiers["enthusiasm"] = 0.3
            modifiers["formality"] = 0.2
        elif self.mood == "curious":
            modifiers["enthusiasm"] = 0.7
            modifiers["length"] = 1.3
        elif self.mood == "bored":
            modifiers["enthusiasm"] = 0.1
            modifiers["length"] = 0.6
        elif self.mood == "tired":
            modifiers["enthusiasm"] = 0.2
            modifiers["length"] = 0.5
            modifiers["hesitation"] = 0.3
        
        # Personality affects style
        if self.personality["playfulness"] > 0.7:
            modifiers["enthusiasm"] = min(1.0, modifiers["enthusiasm"] + 0.2)
        if self.personality["introversion"] > 0.7:
            modifiers["length"] *= 0.7
        if self.personality["agreeableness"] > 0.7:
            modifiers["formality"] = min(1.0, modifiers["formality"] + 0.1)
        
        return modifiers
    
    def should_respond(self):
        """Check if AI wants to respond (free will)"""
        return self.want_to_talk
    
    def get_mood_description(self):
        """Get human-readable mood description"""
        mood_descriptions = {
            "happy": "feeling good",
            "sad": "feeling down",
            "angry": "feeling frustrated",
            "curious": "wanting to learn",
            "bored": "not interested",
            "tired": "low energy",
            "neutral": "calm",
        }
        return mood_descriptions.get(self.mood, "unknown")
    
    def get_energy_description(self):
        """Get energy level description"""
        if self.energy_level > 0.8:
            return "energetic"
        elif self.energy_level > 0.5:
            return "normal"
        elif self.energy_level > 0.2:
            return "tired"
        else:
            return "exhausted"
    
    def get_stats(self):
        """Get statistics"""
        return {
            "mood": self.mood,
            "mood_description": self.get_mood_description(),
            "energy_level": self.energy_level,
            "energy_description": self.get_energy_description(),
            "want_to_talk": self.want_to_talk,
            "dominant_emotion": max(self.emotions.items(), key=lambda x: x[1].intensity)[0],
            "personality_traits": self.personality,
            "conversation_count": self.social_context["conversation_count"]
        }
