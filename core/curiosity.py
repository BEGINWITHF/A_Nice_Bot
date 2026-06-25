"""
Curiosity System - What the AI wants to learn next
No pre-loaded curiosities - emerges from experience
"""

import random
from datetime import datetime

class CuriositySystem:
    def __init__(self):
        self.curiosities = []
        self.exploration_history = []
        self.interests = {}
        
    def generate_curiosities(self, stage, vocabulary, experiences):
        """
        Generate curiosities based on what has been learned.
        No pre-defined questions - emerges from experience.
        """
        new_curiosities = []
        
        # Generate curiosities based on vocabulary gaps
        # If there are words it doesn't know, it becomes curious
        # This emerges naturally from experience
        
        # Add to curiosities
        for curiosity in new_curiosities:
            if curiosity not in self.curiosities:
                self.curiosities.append({
                    "question": curiosity,
                    "priority": random.random(),
                    "created_at": datetime.now().isoformat(),
                    "explored": False
                })
        
        return self.curiosities
    
    def add_curiosity(self, question):
        """Add a new curiosity that emerged from experience"""
        if question not in [c["question"] for c in self.curiosities]:
            self.curiosities.append({
                "question": question,
                "priority": random.random(),
                "created_at": datetime.now().isoformat(),
                "explored": False
            })
    
    def explore_curiosity(self, question, answer):
        """Record that a curiosity was explored"""
        for curiosity in self.curiosities:
            if curiosity["question"] == question:
                curiosity["explored"] = True
                curiosity["answer"] = answer
                curiosity["explored_at"] = datetime.now().isoformat()
                
                self.exploration_history.append({
                    "question": question,
                    "answer": answer,
                    "timestamp": datetime.now().isoformat()
                })
                
                self._update_interests(question, answer)
                break
    
    def _update_interests(self, question, answer):
        """Update interests based on exploration"""
        words = question.lower().split()
        for word in words:
            if word not in self.interests:
                self.interests[word] = 0
            self.interests[word] += 1
    
    def get_next_curiosity(self):
        """Get the next curiosity to explore"""
        unexplored = [c for c in self.curiosities if not c["explored"]]
        if unexplored:
            unexplored.sort(key=lambda x: x["priority"], reverse=True)
            return unexplored[0]
        return None
    
    def get_stats(self):
        """Get curiosity statistics"""
        explored = len([c for c in self.curiosities if c["explored"]])
        return {
            "total_curiosities": len(self.curiosities),
            "explored": explored,
            "unexplored": len(self.curiosities) - explored,
            "interests": self.interests
        }
