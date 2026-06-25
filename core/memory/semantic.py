"""
Semantic Memory - What things mean (like human conceptual knowledge)
Stores learned concepts and their meanings
"""

import json
import os
from datetime import datetime

class SemanticMemory:
    def __init__(self, data_dir="data/baby/semantic"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Knowledge structures
        self.words = {}  # word -> definition
        self.concepts = {}  # concept -> explanation
        self.relationships = {}  # concept -> related concepts
        self.categories = {}  # category -> items in category
        
        self._load_knowledge()
    
    def _load_knowledge(self):
        """Load knowledge from disk"""
        knowledge_file = os.path.join(self.data_dir, "knowledge.json")
        if os.path.exists(knowledge_file):
            try:
                with open(knowledge_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.words = data.get("words", {})
                    self.concepts = data.get("concepts", {})
                    self.relationships = data.get("relationships", {})
                    self.categories = data.get("categories", {})
            except:
                pass
    
    def _save_knowledge(self):
        """Save knowledge to disk"""
        knowledge_file = os.path.join(self.data_dir, "knowledge.json")
        data = {
            "words": self.words,
            "concepts": self.concepts,
            "relationships": self.relationships,
            "categories": self.categories,
            "last_updated": datetime.now().isoformat()
        }
        with open(knowledge_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def learn_word(self, word, definition, context=None):
        """
        Learn what a word means (like a baby learning language)
        
        Args:
            word: The word to learn
            definition: What it means
            context: Where/how it was learned
        """
        self.words[word] = {
            "definition": definition,
            "context": context,
            "learned_at": datetime.now().isoformat(),
            "confidence": 0.5  # Starts uncertain
        }
        self._save_knowledge()
    
    def learn_concept(self, concept, explanation, related_to=None):
        """
        Learn an abstract concept (like understanding "love" or "danger")
        
        Args:
            concept: The concept name
            explanation: What it means
            related_to: Related concepts
        """
        self.concepts[concept] = {
            "explanation": explanation,
            "related_to": related_to or [],
            "learned_at": datetime.now().isoformat(),
            "understanding_level": 0.3  # Starts low
        }
        
        # Add relationships
        if related_to:
            for rel in related_to:
                if rel not in self.relationships:
                    self.relationships[rel] = []
                self.relationships[rel].append(concept)
        
        self._save_knowledge()
    
    def learn_category(self, category, items):
        """
        Learn that things belong to a category (like "animals" = dog, cat, etc.)
        
        Args:
            category: Category name
            items: List of items in category
        """
        self.categories[category] = {
            "items": items,
            "learned_at": datetime.now().isoformat()
        }
        self._save_knowledge()
    
    def strengthen_word(self, word, amount=0.1):
        """Strengthen understanding of a word (more exposure = better understanding)"""
        if word in self.words:
            self.words[word]["confidence"] = min(1.0, self.words[word]["confidence"] + amount)
            self._save_knowledge()
    
    def understand(self, text):
        """
        Try to understand text using learned knowledge
        Returns: dict with understanding level and unknown words
        """
        words = text.lower().split()
        known = []
        unknown = []
        
        for word in words:
            if word in self.words:
                known.append(word)
            else:
                unknown.append(word)
        
        understanding_level = len(known) / len(words) if words else 0
        
        return {
            "understanding_level": understanding_level,
            "known_words": known,
            "unknown_words": unknown
        }
    
    def get_word_definition(self, word):
        """Get definition of a word"""
        if word in self.words:
            return self.words[word]["definition"]
        return None
    
    def get_stats(self):
        """Get statistics about learned knowledge"""
        return {
            "words_learned": len(self.words),
            "concepts_learned": len(self.concepts),
            "categories_learned": len(self.categories),
            "avg_word_confidence": sum(w["confidence"] for w in self.words.values()) / len(self.words) if self.words else 0
        }
