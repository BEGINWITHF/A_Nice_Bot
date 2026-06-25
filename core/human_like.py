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
            "conversation_count": self.social_context["conversation_count"],
            "memories_consolidated": len(self.interaction_memories)
        }
    
    def consolidate_memories(self):
        """
        Memory Consolidation Process
        Like sleep-dependent memory consolidation in the brain
        
        During sleep (or rest periods), memories are:
        1. Reactivated (hippocampal replay)
        2. Transferred to long-term storage (neocortex)
        3. Strengthened based on emotional importance
        4. Weak memories are pruned
        """
        consolidated = 0
        
        # Reactivate and strengthen important memories
        for memory in self.interaction_memories:
            importance = memory.get("importance", 0.5)
            emotional_impact = memory.get("emotional_impact", 0.3)
            
            # Calculate memory strength
            strength = importance * 0.6 + emotional_impact * 0.4
            
            # Strengthen strong memories
            if strength > 0.5:
                memory["strength"] = min(1.0, strength + 0.1)
                consolidated += 1
            
            # Decay weak memories
            if strength < 0.2:
                memory["strength"] *= 0.8
        
        # Prune very weak memories (keeping at least some)
        if len(self.interaction_memories) > 10:
            self.interaction_memories = [
                m for m in self.interaction_memories 
                if m.get("strength", 0.5) > 0.1
            ][:50]  # Keep max 50 strong memories
        
        self._save_state()
        return consolidated
    
    def sleep_like_consolidation(self):
        """
        Sleep-like consolidation process
        Simulates the brain's sleep cycle for memory consolidation
        
        Phases:
        1. Slow-wave sleep: Memory replay
        2. REM sleep: Emotional processing
        3. Sleep spindles: Integration with existing memories
        """
        # Phase 1: Slow-wave sleep (memory replay)
        replayed = self._replay_memories()
        
        # Phase 2: REM sleep (emotional processing)
        emotional_processed = self._process_emotions_during_sleep()
        
        # Phase 3: Sleep spindles (integration)
        integrated = self._integrate_memories()
        
        # Reset energy (like waking up refreshed)
        self.energy_level = min(1.0, self.energy_level + 0.3)
        
        # Update mood after sleep
        self._update_mood()
        
        self._save_state()
        
        return {
            "replayed": replayed,
            "emotional_processed": emotional_processed,
            "integrated": integrated,
            "energy_restored": self.energy_level
        }
    
    def _replay_memories(self):
        """
        Memory Replay during sleep
        Like hippocampal replay during slow-wave sleep
        """
        replayed = 0
        
        # Replay recent memories
        recent_memories = self.interaction_memories[-10:] if self.interaction_memories else []
        
        for memory in recent_memories:
            # Reactivate the memory
            importance = memory.get("importance", 0.5)
            
            # Strengthen based on replay
            if "replay_count" not in memory:
                memory["replay_count"] = 0
            memory["replay_count"] += 1
            
            # Increase strength with each replay
            memory["strength"] = min(1.0, memory.get("strength", 0.5) + 0.02)
            replayed += 1
        
        return replayed
    
    def _process_emotions_during_sleep(self):
        """
        Emotional processing during REM sleep
        Like the amygdala processing emotions during dreams
        """
        processed = 0
        
        # Find emotionally charged memories
        emotional_memories = [
            m for m in self.interaction_memories 
            if m.get("emotional_impact", 0) > 0.5
        ]
        
        for memory in emotional_memories:
            # Process and regulate the emotion
            emotional_impact = memory.get("emotional_impact", 0.5)
            
            # Reduce extreme emotions (emotional regulation)
            if emotional_impact > 0.8:
                memory["emotional_impact"] = emotional_impact * 0.9
            
            # Increase strength of emotionally significant memories
            memory["strength"] = min(1.0, memory.get("strength", 0.5) + 0.05)
            processed += 1
        
        return processed
    
    def _integrate_memories(self):
        """
        Memory integration during sleep spindles
        Like thalamocortical spindles connecting memories
        """
        integrated = 0
        
        # Group similar memories
        memory_groups = {}
        for memory in self.interaction_memories:
            topic = memory.get("topic", "general")
            if topic not in memory_groups:
                memory_groups[topic] = []
            memory_groups[topic].append(memory)
        
        # Integrate memories within groups
        for topic, memories in memory_groups.items():
            if len(memories) > 1:
                # Create a consolidated memory
                avg_importance = sum(m.get("importance", 0.5) for m in memories) / len(memories)
                avg_emotional = sum(m.get("emotional_impact", 0.3) for m in memories) / len(memories)
                
                # Mark memories as integrated
                for memory in memories:
                    memory["integrated"] = True
                    integrated += 1
        
        return integrated
    
    def store_interaction_memory(self, input_text, response, emotional_impact=0.5):
        """
        Store an interaction in memory for later consolidation
        Like encoding new memories during waking hours
        """
        memory = {
            "input": input_text,
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "emotional_impact": emotional_impact,
            "importance": self._calculate_importance(input_text),
            "topic": self._extract_topic(input_text),
            "mood_at_time": self.mood,
            "strength": 0.5,
            "replay_count": 0,
            "integrated": False
        }
        
        self.interaction_memories.append(memory)
        
        # Keep only recent memories in working memory
        if len(self.interaction_memories) > 20:
            self.interaction_memories = self.interaction_memories[-20:]
        
        self._save_state()
        return memory
    
    def _calculate_importance(self, text):
        """Calculate importance of an interaction"""
        importance = 0.3  # Base importance
        
        # Questions are important
        if "?" in text:
            importance += 0.2
        
        # Emotional words increase importance
        emotional_words = ["love", "hate", "fear", "happy", "sad", "angry", "excited"]
        for word in emotional_words:
            if word in text.lower():
                importance += 0.1
        
        # Longer inputs are more important
        if len(text) > 50:
            importance += 0.1
        
        return min(1.0, importance)
    
    def _extract_topic(self, text):
        """Extract topic from text for memory organization"""
        text_lower = text.lower()
        
        # Simple topic extraction
        if any(word in text_lower for word in ["hello", "hi", "hey"]):
            return "greeting"
        elif any(word in text_lower for word in ["how", "what", "why", "when", "where"]):
            return "question"
        elif any(word in text_lower for word in ["play", "game", "fun"]):
            return "play"
        elif any(word in text_lower for word in ["love", "like", "feel"]):
            return "emotional"
        
        return "general"
