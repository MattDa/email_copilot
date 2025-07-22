# app/agents/tool_using_agent.py
from typing import List, Dict, Any, Union
from services.chroma_service import ChromaService
import re


class ToolUsingAgent:
    def __init__(self, chroma_service: ChromaService):
        self.chroma_service = chroma_service

    async def execute_query(self, query: Union[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute a query against the email database"""

        # Parse query to extract search parameters
        search_params = self._parse_query(query)

        # Execute search in ChromaDB
        results = await self.chroma_service.search(
            query_text=search_params.get("text", query if isinstance(query, str) else ""),
            n_results=search_params.get("limit", 20),
            where=search_params.get("filters")
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

    def _parse_query(self, query: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Parse query to build text and metadata filters for Chroma"""

        params: Dict[str, Any] = {
            "text": "",
            "limit": 20,
            "filters": {},
        }

        if isinstance(query, dict):
            params["text"] = query.get("text", "")
            if "limit" in query:
                params["limit"] = query["limit"]
            sender = query.get("sender")
            if sender:
                params["filters"]["sender"] = sender
            start = query.get("start_date")
            end = query.get("end_date")
            if start or end:
                date_filter: Dict[str, Any] = {}
                if start:
                    date_filter["$gte"] = start
                if end:
                    date_filter["$lte"] = end
                params["filters"]["date"] = date_filter
        else:
            params["text"] = query
            q_lower = query.lower()

            if any(word in q_lower for word in ["all", "every", "list", "show me"]):
                params["limit"] = 20
<<<<<<< ours

=======
>>>>>>> theirs
            elif any(word in q_lower for word in ["specific", "exact", "particular"]):
                params["limit"] = 15

            sender_match = re.search(r"from[: ]+(\S+@\S+)", q_lower)
            if sender_match:
                params["filters"]["sender"] = sender_match.group(1)

            after_match = re.search(r"(after|since)[: ]+(\d{4}-\d{2}-\d{2})", q_lower)
            if after_match:
                params.setdefault("filters", {}).setdefault("date", {})["$gte"] = after_match.group(2)

            before_match = re.search(r"before[: ]+(\d{4}-\d{2}-\d{2})", q_lower)
            if before_match:
                params.setdefault("filters", {}).setdefault("date", {})["$lte"] = before_match.group(1)

        params["limit"] = min(max(params.get("limit", 20), 1), 20)
        if not params["filters"]:
            params["filters"] = None

        return params
