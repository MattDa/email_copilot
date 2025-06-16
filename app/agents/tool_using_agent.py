# app/agents/tool_using_agent.py
from typing import List, Dict, Any
from services.chroma_service import ChromaService
import re


class ToolUsingAgent:
    def __init__(self, chroma_service: ChromaService):
        self.chroma_service = chroma_service

    async def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute a query against the email database"""

        # Parse query to extract search parameters
        search_params = self._parse_query(query)

        # Execute search in ChromaDB
        results = await self.chroma_service.search(
            query_text=search_params.get('text', query),
            n_results=search_params.get('limit', 25),
            where=search_params.get('filters', None)  # Pass None instead of empty dict
        )

        # Ensure results are properly formatted
        formatted_results = []
        for result in results:
            # Make sure all required fields are present
            formatted_result = {
                'id': result.get('id', ''),
                'content': result.get('content', ''),
                'subject': result.get('subject', ''),
                'sender': result.get('sender', ''),
                'recipient': result.get('recipient', ''),
                'date': result.get('date', ''),
                'score': result.get('score', 0.0)
            }
            formatted_results.append(formatted_result)

        return formatted_results

    def _parse_query(self, query: str) -> Dict[str, Any]:
        """Parse query to extract search parameters"""
        params = {
            'text': query,
            'limit': 25,
            'filters': None  # Start with None
        }

        # For now, disabled complex filtering and rely on semantic search
        # ChromaDB's where clause has specific requirements that are causing issues

        # Extract specific search patterns but don't use them as filters
        query_lower = query.lower()

        # Adjust result limit based on query characteristics
        if any(word in query_lower for word in ['all', 'every', 'list', 'show me']):
            params['limit'] = 50  # Broader queries might need more results
        elif any(word in query_lower for word in ['specific', 'exact', 'particular']):
            params['limit'] = 15  # More specific queries need fewer results

        # For complex filtering, we'll handle it post-search in the results
        return params