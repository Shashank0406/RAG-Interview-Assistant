"""
Conversation Memory System for RAG
Manages chat history and context for better conversations
"""

import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class ConversationMessage:
    """Represents a single message in the conversation"""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationMessage':
        """Create from dictionary"""
        data_copy = data.copy()
        data_copy['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data_copy)


@dataclass
class ConversationContext:
    """Context information for the current conversation"""
    user_id: str
    session_id: str
    topic: Optional[str] = None
    tags: List[str] = None
    summary: Optional[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class ConversationMemory:
    """Manages conversation history and context"""

    def __init__(self,
                 max_messages: int = 50,
                 max_age_hours: int = 24,
                 memory_file: Optional[str] = None):
        """
        Initialize conversation memory

        Args:
            max_messages: Maximum number of messages to keep in memory
            max_age_hours: Maximum age of messages in hours
            memory_file: File path to persist conversations (optional)
        """
        self.max_messages = max_messages
        self.max_age = timedelta(hours=max_age_hours)
        self.memory_file = memory_file

        # In-memory storage: session_id -> deque of messages
        self.conversations: Dict[str, deque] = {}
        self.contexts: Dict[str, ConversationContext] = {}

        # Load persisted conversations if file exists
        if memory_file:
            self._load_from_file()

    def add_message(self,
                   session_id: str,
                   role: str,
                   content: str,
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a message to the conversation"""
        message = ConversationMessage(
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )

        # Initialize conversation if it doesn't exist
        if session_id not in self.conversations:
            self.conversations[session_id] = deque(maxlen=self.max_messages)

        # Add message
        self.conversations[session_id].append(message)

        # Auto-save if memory file is configured
        if self.memory_file:
            self._save_to_file()

        logger.debug(f"Added {role} message to session {session_id}")

    def get_recent_messages(self,
                           session_id: str,
                           limit: Optional[int] = None,
                           max_age: Optional[timedelta] = None) -> List[ConversationMessage]:
        """Get recent messages from a conversation"""
        if session_id not in self.conversations:
            return []

        messages = list(self.conversations[session_id])

        # Filter by age if specified
        if max_age:
            cutoff_time = datetime.now() - max_age
            messages = [msg for msg in messages if msg.timestamp > cutoff_time]

        # Apply limit
        if limit:
            messages = messages[-limit:]

        return messages

    def get_conversation_history(self,
                                session_id: str,
                                include_system: bool = False) -> List[Dict[str, str]]:
        """Get conversation history in format suitable for LLM APIs"""
        messages = self.get_recent_messages(session_id)

        history = []
        for msg in messages:
            if msg.role == "system" and not include_system:
                continue
            history.append({
                "role": msg.role,
                "content": msg.content
            })

        return history

    def get_context_summary(self, session_id: str) -> Optional[str]:
        """Get the context summary for a conversation"""
        context = self.contexts.get(session_id)
        return context.summary if context else None

    def set_context(self,
                   session_id: str,
                   user_id: str,
                   topic: Optional[str] = None,
                   tags: Optional[List[str]] = None) -> None:
        """Set context information for a conversation"""
        self.contexts[session_id] = ConversationContext(
            user_id=user_id,
            session_id=session_id,
            topic=topic,
            tags=tags or []
        )

    def update_context_summary(self, session_id: str, summary: str) -> None:
        """Update the context summary for a conversation"""
        if session_id in self.contexts:
            self.contexts[session_id].summary = summary

    def search_conversations(self,
                           query: str,
                           user_id: Optional[str] = None,
                           limit: int = 10) -> List[Dict[str, Any]]:
        """Search through conversation history"""
        results = []

        for session_id, messages in self.conversations.items():
            context = self.contexts.get(session_id)

            # Filter by user if specified
            if user_id and context and context.user_id != user_id:
                continue

            # Search through messages
            for msg in messages:
                if query.lower() in msg.content.lower():
                    results.append({
                        "session_id": session_id,
                        "user_id": context.user_id if context else None,
                        "message": msg.to_dict(),
                        "topic": context.topic if context else None,
                        "tags": context.tags if context else []
                    })

                    if len(results) >= limit:
                        break

            if len(results) >= limit:
                break

        return results

    def clear_conversation(self, session_id: str) -> None:
        """Clear all messages for a conversation"""
        if session_id in self.conversations:
            self.conversations[session_id].clear()

        if session_id in self.contexts:
            del self.contexts[session_id]

        if self.memory_file:
            self._save_to_file()

        logger.info(f"Cleared conversation for session {session_id}")

    def get_conversation_stats(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about conversations"""
        if session_id:
            # Stats for specific session
            messages = self.get_recent_messages(session_id)
            context = self.contexts.get(session_id)

            return {
                "session_id": session_id,
                "message_count": len(messages),
                "user_id": context.user_id if context else None,
                "topic": context.topic if context else None,
                "tags": context.tags if context else [],
                "oldest_message": messages[0].timestamp if messages else None,
                "newest_message": messages[-1].timestamp if messages else None
            }
        else:
            # Overall stats
            total_messages = sum(len(msgs) for msgs in self.conversations.values())
            active_sessions = len(self.conversations)

            return {
                "total_sessions": len(self.conversations),
                "active_sessions": active_sessions,
                "total_messages": total_messages,
                "average_messages_per_session": total_messages / active_sessions if active_sessions > 0 else 0
            }

    def cleanup_old_messages(self) -> int:
        """Remove messages older than max_age"""
        removed_count = 0
        cutoff_time = datetime.now() - self.max_age

        for session_id, messages in self.conversations.items():
            original_length = len(messages)
            # Keep only messages newer than cutoff
            filtered_messages = deque(
                (msg for msg in messages if msg.timestamp > cutoff_time),
                maxlen=self.max_messages
            )
            self.conversations[session_id] = filtered_messages
            removed_count += original_length - len(filtered_messages)

        if removed_count > 0 and self.memory_file:
            self._save_to_file()

        logger.info(f"Cleaned up {removed_count} old messages")
        return removed_count

    def _save_to_file(self) -> None:
        """Save conversations to file"""
        if not self.memory_file:
            return

        try:
            data = {
                "conversations": {},
                "contexts": {}
            }

            # Serialize conversations
            for session_id, messages in self.conversations.items():
                data["conversations"][session_id] = [msg.to_dict() for msg in messages]

            # Serialize contexts
            for session_id, context in self.contexts.items():
                data["contexts"][session_id] = asdict(context)

            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Error saving conversations to file: {e}")

    def _load_from_file(self) -> None:
        """Load conversations from file"""
        if not self.memory_file or not os.path.exists(self.memory_file):
            return

        try:
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Deserialize conversations
            for session_id, messages_data in data.get("conversations", {}).items():
                messages = deque(
                    [ConversationMessage.from_dict(msg_data) for msg_data in messages_data],
                    maxlen=self.max_messages
                )
                self.conversations[session_id] = messages

            # Deserialize contexts
            for session_id, context_data in data.get("contexts", {}).items():
                self.contexts[session_id] = ConversationContext(**context_data)

            logger.info(f"Loaded conversations from {self.memory_file}")

        except Exception as e:
            logger.error(f"Error loading conversations from file: {e}")


class ConversationManager:
    """High-level interface for conversation management"""

    def __init__(self, memory: ConversationMemory):
        self.memory = memory

    def start_conversation(self,
                          user_id: str,
                          session_id: Optional[str] = None,
                          topic: Optional[str] = None) -> str:
        """Start a new conversation"""
        if not session_id:
            session_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.memory.set_context(session_id, user_id, topic)

        # Add welcome message
        self.memory.add_message(
            session_id,
            "system",
            f"Starting new conversation with user {user_id}" + (f" on topic: {topic}" if topic else "")
        )

        return session_id

    def add_user_message(self, session_id: str, message: str) -> None:
        """Add a user message to the conversation"""
        self.memory.add_message(session_id, "user", message)

    def add_assistant_message(self, session_id: str, message: str, metadata: Optional[Dict] = None) -> None:
        """Add an assistant message to the conversation"""
        self.memory.add_message(session_id, "assistant", message, metadata)

    def get_formatted_history(self, session_id: str, max_messages: int = 10) -> str:
        """Get formatted conversation history for context"""
        messages = self.memory.get_recent_messages(session_id, limit=max_messages)

        if not messages:
            return "No previous conversation history."

        formatted = []
        for msg in messages[-max_messages:]:  # Get last N messages
            role = "User" if msg.role == "user" else "Assistant"
            timestamp = msg.timestamp.strftime("%H:%M")
            formatted.append(f"[{timestamp}] {role}: {msg.content}")

        return "\n".join(formatted)

    def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a summary of the conversation"""
        stats = self.memory.get_conversation_stats(session_id)
        recent_messages = self.memory.get_recent_messages(session_id, limit=5)

        return {
            "stats": stats,
            "recent_activity": [msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                              for msg in recent_messages],
            "context": self.memory.get_context_summary(session_id)
        }


# Convenience functions
def create_memory_system(memory_file: Optional[str] = None) -> ConversationManager:
    """Create a conversation memory system"""
    memory = ConversationMemory(memory_file=memory_file)
    return ConversationManager(memory)