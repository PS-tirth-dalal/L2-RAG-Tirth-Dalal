# memory.py
# Conversation Memory -- persists chat history and user context to a JSON file.
# Provides session tracking, topic extraction, and recent context retrieval.

import json
import os
from datetime import datetime
from typing import List, Dict, Optional


class ConversationMemory:
    """Persists conversation history and user preferences to a local JSON file.

    Attributes:
        filepath: Path to the JSON file used for persistence.
        data: The in-memory representation of stored conversation data.
    """

    DEFAULT_DATA = {
        "user_name": None,
        "session_count": 0,
        "last_session": None,
        "exchanges": [],
        "topics_discussed": [],
    }

    def __init__(self, filepath: str = "memory.json") -> None:
        self.filepath = filepath
        self.data: Dict = {}
        self.load()

    # ── Persistence ──────────────────────────────

    def load(self) -> Dict:
        """Load conversation data from the JSON file. Creates defaults if missing."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.data = dict(self.DEFAULT_DATA)
        else:
            self.data = dict(self.DEFAULT_DATA)
        return self.data

    def save(self) -> None:
        """Write current conversation data to the JSON file."""
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    # ── Session management ───────────────────────

    def start_session(self) -> None:
        """Increment session count and record the current timestamp."""
        self.data["session_count"] = self.data.get("session_count", 0) + 1
        self.data["last_session"] = datetime.now().isoformat()

    def is_returning_user(self) -> bool:
        """Return True if the user has chatted at least once before."""
        return self.data.get("session_count", 0) > 1

    # ── Exchange tracking ────────────────────────

    def add_exchange(self, user_msg: str, agent_response: str) -> None:
        """Append a user/agent exchange and auto-extract topics."""
        exchange = {
            "timestamp": datetime.now().isoformat(),
            "user": user_msg,
            "agent": agent_response,
        }
        self.data.setdefault("exchanges", []).append(exchange)

        # Auto-extract topics from user message (simple keyword matching)
        self._extract_topics(user_msg)
        self.save()

    def get_recent_context(self, n: int = 5) -> List[Dict]:
        """Return the last *n* exchanges for context injection."""
        exchanges = self.data.get("exchanges", [])
        return exchanges[-n:] if exchanges else []

    # ── Topic extraction ─────────────────────────

    TOPIC_KEYWORDS = {
        "weather": ["weather", "temperature", "rain", "sunny", "forecast", "climate"],
        "books": ["book", "read", "novel", "author", "library", "literature"],
        "jokes": ["joke", "funny", "laugh", "humor", "comedy"],
        "dogs": ["dog", "puppy", "pet", "canine"],
        "trivia": ["trivia", "quiz", "question", "knowledge", "test"],
        "movies": ["movie", "film", "watch", "cinema"],
        "food": ["recipe", "cook", "food", "meal", "eat", "brunch", "dinner"],
    }

    def _extract_topics(self, text: str) -> None:
        """Scan text for known topic keywords and add new ones to the list."""
        text_lower = text.lower()
        topics = self.data.setdefault("topics_discussed", [])
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            if topic not in topics and any(kw in text_lower for kw in keywords):
                topics.append(topic)

    # ── Summary for prompt injection ─────────────

    def get_summary(self) -> str:
        """Return a human-readable summary suitable for injecting into the system prompt."""
        sessions = self.data.get("session_count", 0)
        if sessions <= 1:
            return ""

        parts = [f"The user has chatted {sessions} times before."]

        topics = self.data.get("topics_discussed", [])
        if topics:
            parts.append(f"Topics they have discussed: {', '.join(topics)}.")

        recent = self.get_recent_context(3)
        if recent:
            last_msg = recent[-1].get("user", "")
            if last_msg:
                parts.append(f'Their most recent message was: "{last_msg}"')

        name = self.data.get("user_name")
        if name:
            parts.insert(0, f"The user's name is {name}.")

        return " ".join(parts)

    # ── Reset ────────────────────────────────────

    def clear(self) -> None:
        """Reset all stored memory to defaults."""
        self.data = dict(self.DEFAULT_DATA)
        self.save()

    def __repr__(self) -> str:
        sessions = self.data.get("session_count", 0)
        exchanges = len(self.data.get("exchanges", []))
        return f"ConversationMemory(sessions={sessions}, exchanges={exchanges})"
