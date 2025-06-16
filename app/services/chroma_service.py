# app/services/chroma_service.py
import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer


class ChromaService:
    def __init__(self):
        self.client = None
        self.collection = None
        self.embedder = None

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

        # Get the email body content
        body_content = email_data.get('body', '')
        subject = email_data.get('subject', '')

        # Create content for embedding (subject + body)
        embedding_content = f"{subject} {body_content}"

        # Debug print
        print(f"Adding email to ChromaDB: Subject='{subject}', Body length={len(body_content)}")

        # Create embedding
        embedding = self.embedder.encode(embedding_content).tolist()

        # Generate unique ID if not provided
        email_id = email_data.get('message_id')
        if not email_id:
            try:
                existing = self.collection.get()
                email_id = f"email_{len(existing['ids'])}"
            except:
                email_id = "email_0"

        # Store the full body content in the document field
        # This is crucial - the document field is what gets returned in searches
        document_content = body_content if body_content else subject

        # Add to collection
        self.collection.add(
            embeddings=[embedding],
            documents=[document_content],  # This should contain the email body
            metadatas=[{
                'subject': subject,
                'sender': email_data.get('sender', ''),
                'recipient': email_data.get('recipient', ''),
                'date': email_data.get('date', ''),
                'message_id': email_id,
                'body_length': len(body_content)  # Store body length for debugging
            }],
            ids=[email_id]
        )

        print(f"Successfully added email {email_id} to ChromaDB")

    async def search(
            self,
            query_text: str,
            n_results: int = 10,
            where: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Search emails in ChromaDB"""

        if not self.collection:
            await self.initialize()

        # Create query embedding
        query_embedding = self.embedder.encode(query_text).tolist()

        # Search
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
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