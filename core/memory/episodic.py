"""
Episodic Memory - What happened (like human autobiographical memory)
Stores specific experiences and events
"""

import json
import os
from datetime import datetime

class EpisodicMemory:
    def __init__(self, data_dir="data/baby/episodes"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.episodes = []
        self._load_episodes()
    
    def _load_episodes(self):
        """Load episodes from disk"""
        episodes_file = os.path.join(self.data_dir, "episodes.json")
        if os.path.exists(episodes_file):
            try:
                with open(episodes_file, "r", encoding="utf-8") as f:
                    self.episodes = json.load(f)
            except:
                self.episodes = []
    
    def _save_episodes(self):
        """Save episodes to disk"""
        episodes_file = os.path.join(self.data_dir, "episodes.json")
        with open(episodes_file, "w", encoding="utf-8") as f:
            json.dump(self.episodes, f, indent=2, ensure_ascii=False)
    
    def record(self, event_type, description, emotional_weight=0.5, context=None):
        """
        Record a new episode (like a baby experiencing something)
        
        Args:
            event_type: "sensory", "social", "learning", "discovery"
            description: What happened
            emotional_weight: 0.0 (neutral) to 1.0 (very emotional)
            context: Additional context
        """
        episode = {
            "id": len(self.episodes),
            "type": event_type,
            "description": description,
            "emotional_weight": emotional_weight,
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "rehearsals": 0  # How many times recalled
        }
        
        self.episodes.append(episode)
        self._save_episodes()
        
        return episode
    
    def recall_recent(self, n=5):
        """Recall most recent episodes"""
        return self.episodes[-n:] if self.episodes else []
    
    def recall_by_type(self, event_type, limit=10):
        """Recall episodes of a specific type"""
        filtered = [e for e in self.episodes if e["type"] == event_type]
        return filtered[-limit:] if filtered else []
    
    def recall_emotional(self, min_weight=0.7):
        """Recall emotionally significant episodes"""
        return [e for e in self.episodes if e["emotional_weight"] >= min_weight]
    
    def recall_with_person(self, person_name):
        """Recall episodes involving a specific person"""
        return [e for e in self.episodes if person_name in str(e.get("context", ""))]
    
    def forget(self, episode_id):
        """Forget an episode (like human forgetting)"""
        if 0 <= episode_id < len(self.episodes):
            self.episodes.pop(episode_id)
            self._save_episodes()
            return True
        return False
    
    def get_summary(self):
        """Get summary of all memories"""
        if not self.episodes:
            return "No memories yet."
        
        types = {}
        for ep in self.episodes:
            t = ep["type"]
            types[t] = types.get(t, 0) + 1
        
        summary = f"Total memories: {len(self.episodes)}\n"
        for t, count in types.items():
            summary += f"  - {t}: {count}\n"
        
        return summary
