"""
Procedural Memory - How to do things (like human muscle memory)
Stores learned behaviors and skills
"""

import json
import os
from datetime import datetime

class ProceduralMemory:
    def __init__(self, data_dir="data/baby/procedural"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Skills and behaviors
        self.skills = {}  # skill -> proficiency
        self.behaviors = {}  # situation -> response
        self.habits = []  # frequently repeated actions
        
        self._load_skills()
    
    def _load_skills(self):
        """Load skills from disk"""
        skills_file = os.path.join(self.data_dir, "skills.json")
        if os.path.exists(skills_file):
            try:
                with open(skills_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.skills = data.get("skills", {})
                    self.behaviors = data.get("behaviors", {})
                    self.habits = data.get("habits", [])
            except:
                pass
    
    def _save_skills(self):
        """Save skills to disk"""
        skills_file = os.path.join(self.data_dir, "skills.json")
        data = {
            "skills": self.skills,
            "behaviors": self.behaviors,
            "habits": self.habits,
            "last_updated": datetime.now().isoformat()
        }
        with open(skills_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def learn_skill(self, skill_name, description, steps=None):
        """
        Learn a new skill (like a baby learning to grasp objects)
        
        Args:
            skill_name: Name of the skill
            description: What the skill does
            steps: Steps to perform the skill
        """
        self.skills[skill_name] = {
            "description": description,
            "steps": steps or [],
            "proficiency": 0.1,  # Starts low
            "attempts": 0,
            "successes": 0,
            "learned_at": datetime.now().isoformat()
        }
        self._save_skills()
    
    def practice_skill(self, skill_name, success=True):
        """
        Practice a skill (like a baby practicing walking)
        
        Args:
            skill_name: Skill to practice
            success: Whether the attempt was successful
        """
        if skill_name in self.skills:
            skill = self.skills[skill_name]
            skill["attempts"] += 1
            if success:
                skill["successes"] += 1
                skill["proficiency"] = min(1.0, skill["proficiency"] + 0.05)
            else:
                skill["proficiency"] = max(0.0, skill["proficiency"] - 0.02)
            
            self._save_skills()
    
    def learn_behavior(self, situation, response):
        """
        Learn how to respond to a situation (like learning social cues)
        
        Args:
            situation: The situation or trigger
            response: How to respond
        """
        self.behaviors[situation] = {
            "response": response,
            "learned_at": datetime.now().isoformat(),
            "times_used": 0
        }
        self._save_skills()
    
    def respond_to_situation(self, situation):
        """Try to respond to a learned situation"""
        if situation in self.behaviors:
            behavior = self.behaviors[situation]
            behavior["times_used"] += 1
            self._save_skills()
            return behavior["response"]
        return None
    
    def add_habit(self, action):
        """
        Add a habit (something done frequently)
        
        Args:
            action: The habitual action
        """
        if action not in self.habits:
            self.habits.append(action)
            self._save_skills()
    
    def get_skill_proficiency(self, skill_name):
        """Get proficiency level for a skill"""
        if skill_name in self.skills:
            return self.skills[skill_name]["proficiency"]
        return 0.0
    
    def get_stats(self):
        """Get statistics about learned skills"""
        total_proficiency = sum(s["proficiency"] for s in self.skills.values())
        avg_proficiency = total_proficiency / len(self.skills) if self.skills else 0
        
        return {
            "skills_learned": len(self.skills),
            "behaviors_learned": len(self.behaviors),
            "habits_formed": len(self.habits),
            "avg_proficiency": avg_proficiency
        }
