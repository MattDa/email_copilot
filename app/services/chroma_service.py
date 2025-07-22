# app/services/chroma_service.py
import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import math
import tiktoken
from sentence_transformers import SentenceTransformer


class ChromaService:
    def __init__(self):
        self.client = None
        self.collection = None
        self.embedder = None
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _chunk_text(
        self, text: str, max_tokens: int = 1000, overlap_ratio: float = 0.15
    ) -> List[str]:
        """Chunk text into pieces <= max_tokens with specified overlap."""
        tokens = self.tokenizer.encode(text)
        total = len(tokens)
        if total <= max_tokens:
            return [text]

        num_chunks = math.ceil(total / max_tokens)
        chunk_size = min(max_tokens, math.ceil(total / num_chunks))
        step = max(1, int(chunk_size * (1 - overlap_ratio)))

        chunks = []
        start = 0
        while start < total:
            chunk_tokens = tokens[start : start + chunk_size]
            chunks.append(self.tokenizer.decode(chunk_tokens))
            if start + chunk_size >= total:
                break
            start += step

        return chunks

    async def initialize(self):
        """Initialize ChromaDB client and collection"""
        chroma_host = os.getenv("CHROMA_HOST", "localhost")
        chroma_port = int(os.getenv("CHROMA_PORT", 8000))

        self.client = chromadb.HttpClient(
            host=chroma_host,
            port=chroma_port
        )

        # Initialize embedding model
        self.embedder = SentenceTransformer('jinaai/jina-embeddings-v2-base-en')

        # Get or create collection
        try:
            self.collection = self.client.get_collection("emails")
        except:
            self.collection = self.client.create_collection(
                name="emails",
                metadata={"description": "Email content and metadata"}
            )

    async def add_email(self, email_data: Dict[str, Any]):
        """Add email to ChromaDB"""
        if not self.collection:
            await self.initialize()

        body_content = email_data.get("body", "")
        subject = email_data.get("subject", "")

        print(
            f"Adding email to ChromaDB: Subject='{subject}', Body length={len(body_content)}"
        )

        # Determine chunks for the body content
        chunks = self._chunk_text(body_content)

        email_id = email_data.get("message_id")
        if not email_id:
            try:
                existing = self.collection.get()
                email_id = f"email_{len(existing['ids'])}"
            except Exception:
                email_id = "email_0"

        for idx, chunk in enumerate(chunks, 1):
            embedding_content = f"{subject} {chunk}"
            embedding = self.embedder.encode(embedding_content).tolist()

            chunk_id = f"{email_id}_chunk{idx}"
            self.collection.add(
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{
                    "subject": subject,
                    "sender": email_data.get("sender", ""),
                    "recipient": email_data.get("recipient", ""),
                    "date": email_data.get("date", ""),
                    "message_id": email_id,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "body_length": len(chunk),
                }],
                ids=[chunk_id],
            )

        print(f"Successfully added email {email_id} with {len(chunks)} chunks to ChromaDB")

    async def search(
        self,
        query_text: str,
        n_results: int = 10,
        where: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Search emails in ChromaDB"""

        if not self.collection:
            await self.initialize()

        n_results = min(max(n_results, 1), 20)

        # Create query embedding
        query_embedding = self.embedder.encode(query_text).tolist()

        # Search
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
                where=where,
            )
        except Exception as e:
            print(f"ChromaDB query error: {e}")
            return []

        # Format results with all necessary fields
        formatted_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                metadata = results['metadatas'][0][i]
                document_content = results['documents'][0][i]

                # Debug print
                print(
                    f"Retrieved email {i}: Subject='{metadata.get('subject')}', Content length={len(document_content)}")

                formatted_results.append({
                    'id': results['ids'][0][i],
                    'content': document_content,  # This should be the email body
                    'subject': metadata.get('subject', 'No Subject'),
                    'sender': metadata.get('sender', 'Unknown Sender'),
                    'recipient': metadata.get('recipient', 'Unknown Recipient'),
                    'date': metadata.get('date', 'Unknown Date'),
                    'score': 1 - results['distances'][0][i],
                    'body_length': metadata.get('body_length', 0)  # For debugging
                })

        return formatted_results

    async def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection"""
        if not self.collection:
            await self.initialize()

        try:
            data = self.collection.get()
            return {
                'total_emails': len(data['ids']),
                'collection_name': 'emails'
            }
        except Exception as e:
            return {
                'total_emails': 0,
                'collection_name': 'emails',
                'error': str(e)
            }
