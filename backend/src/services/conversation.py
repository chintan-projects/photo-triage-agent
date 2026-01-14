"""Conversation state management for chat sessions."""
import uuid
from dataclasses import dataclass, field
from datetime import datetime

import aiosqlite
import structlog

logger = structlog.get_logger()


@dataclass
class Message:
    """A single message in a conversation."""

    role: str  # 'user' or 'assistant'
    content: str
    photo_ids: list[int] | None = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Conversation:
    """A conversation session with message history."""

    id: str
    messages: list[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def add_message(self, role: str, content: str, photo_ids: list[int] | None = None):
        """Add a message to the conversation."""
        self.messages.append(Message(role=role, content=content, photo_ids=photo_ids))
        self.updated_at = datetime.now()

    def get_history(self, limit: int = 10) -> list[dict]:
        """Get recent message history as dicts."""
        recent = self.messages[-limit:]
        return [
            {"role": m.role, "content": m.content, "photo_ids": m.photo_ids}
            for m in recent
        ]


class ConversationService:
    """Manages conversation sessions with persistence."""

    def __init__(self, db: aiosqlite.Connection):
        self.db = db
        self._cache: dict[str, Conversation] = {}

    async def create(self) -> Conversation:
        """Create a new conversation."""
        conv_id = str(uuid.uuid4())

        await self.db.execute(
            "INSERT INTO conversations (id) VALUES (?)",
            (conv_id,)
        )
        await self.db.commit()

        conv = Conversation(id=conv_id)
        self._cache[conv_id] = conv

        logger.info("conversation_created", conversation_id=conv_id)
        return conv

    async def get(self, conversation_id: str) -> Conversation | None:
        """Get conversation by ID, loading from DB if needed."""
        # Check cache first
        if conversation_id in self._cache:
            return self._cache[conversation_id]

        # Load from database
        cursor = await self.db.execute(
            "SELECT id, created_at, updated_at FROM conversations WHERE id = ?",
            (conversation_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return None

        conv = Conversation(
            id=row[0],
            created_at=row[1] if row[1] else datetime.now(),
            updated_at=row[2] if row[2] else datetime.now(),
        )

        # Load messages
        cursor = await self.db.execute(
            """
            SELECT role, content, photo_ids, created_at
            FROM conversation_messages
            WHERE conversation_id = ?
            ORDER BY created_at ASC
            """,
            (conversation_id,)
        )
        rows = await cursor.fetchall()
        for msg_row in rows:
            import json
            photo_ids = json.loads(msg_row[2]) if msg_row[2] else None
            conv.messages.append(Message(
                role=msg_row[0],
                content=msg_row[1],
                photo_ids=photo_ids,
                timestamp=msg_row[3] if msg_row[3] else datetime.now(),
            ))

        self._cache[conversation_id] = conv
        return conv

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        photo_ids: list[int] | None = None,
    ) -> Message:
        """Add a message to a conversation."""
        conv = await self.get(conversation_id)
        if not conv:
            raise ValueError(f"Conversation not found: {conversation_id}")

        import json
        photo_ids_json = json.dumps(photo_ids) if photo_ids else None

        await self.db.execute(
            """
            INSERT INTO conversation_messages (conversation_id, role, content, photo_ids)
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, role, content, photo_ids_json)
        )

        await self.db.execute(
            "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (conversation_id,)
        )
        await self.db.commit()

        message = Message(role=role, content=content, photo_ids=photo_ids)
        conv.add_message(role, content, photo_ids)

        return message

    async def get_or_create(self, conversation_id: str | None) -> Conversation:
        """Get existing conversation or create new one."""
        if conversation_id:
            conv = await self.get(conversation_id)
            if conv:
                return conv

        return await self.create()
