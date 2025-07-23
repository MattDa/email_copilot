# app/agents/plan_execute_agent.py
import json
import tiktoken
import os
import re
from typing import List, Dict, Any
from langchain.schema import BaseMessage, HumanMessage, AIMessage

from services.chroma_service import ChromaService
from services.llm_service import LLMService
from agents.tool_using_agent import ToolUsingAgent


class PlanExecuteAgent:
    def __init__(self, chroma_service: ChromaService, llm_service: LLMService):
        self.chroma_service = chroma_service
        self.llm_service = llm_service
        self.tool_agent = ToolUsingAgent(chroma_service)
        self.max_context_tokens = int(os.getenv("MAX_CONTEXT_TOKENS", 999))
        self.max_queries = int(os.getenv("MAX_QUERIES", 10))
        self.encoder = tiktoken.get_encoding("cl100k_base")

    async def execute(self, user_prompt: str) -> Dict[str, Any]:
        """Execute the plan and execute workflow"""

        # Step 1: Create initial plan
        plan = await self._create_plan(user_prompt)

        chain_of_thought = [{
            "step": 1,
            "type": "plan_creation",
            "content": f"Created search plan: {plan[:400]}..."
        }]

        # Step 2: Execute initial broad query
        initial_query = await self._generate_initial_query(user_prompt, plan)
        chain_of_thought.append({
            "step": 2,
            "type": "initial_query",
            "content": f"Generated initial query: {json.dumps(initial_query)}"
        })

        # Get initial results
        email_results = await self.tool_agent.execute_query(initial_query)

        chain_of_thought.append({
            "step": 3,
            "type": "initial_results",
            "content": f"Retrieved {len(email_results)} initial chunks"
        })

        if not email_results:
            return {
                "response": "I couldn't find any relevant emails matching your query. Please try rephrasing your question or check if the emails have been properly uploaded.",
                "chain_of_thought": chain_of_thought,
                "context_tokens_used": 0,
                "queries_executed": 1
            }

        # Step 3: Iteratively refine and filter results
        filtered_results = email_results
        current_query_num = 1

        # Check token count of current results
        current_tokens = self._count_tokens_from_emails(filtered_results)

        while (current_query_num <= self.max_queries or
               current_tokens <= self.max_context_tokens):
            # Check token count of current results
            current_tokens = self._count_tokens_from_emails(filtered_results)

            chain_of_thought.append({
                "step": current_query_num + 3,
                "type": "token_check",
                "content": f"Current context tokens: {current_tokens}/{self.max_context_tokens} with {len(filtered_results)} chunks"
            })

            # If within limits, we're good to proceed
            if current_tokens <= self.max_context_tokens:
                break

            # If too large, first try metadata-based filtering
            pre_filter_len = len(filtered_results)
            filtered_results = self.__metadata_filter(filtered_results, user_prompt, plan)

            if len(filtered_results) < pre_filter_len:
                chain_of_thought.append({
                    "step": current_query_num + 3,
                    "type": "metadata_filtering",
                    "content": f"Applied metadata filtering, reduced to {len(filtered_results)} chunks"
                })

            current_tokens = self._count_tokens_from_emails(filtered_results)
            if current_tokens <= self.max_context_tokens:
                break

            if current_query_num >= self.max_queries:
                # Last attempt - filter as much as possible

                filtered_results = self.__metadata_filter(filtered_results, user_prompt, plan)
                chain_of_thought.append({
                    "step": current_query_num + 3,
                    "type": "metadata_filtering",
                    "content": f"Applied metadata filtering, reduced to {len(filtered_results)} chunks"
                })
                break

            # Refine query to get more targeted results
            refined_query = await self._refine_query(user_prompt, plan, filtered_results, current_query_num)
            chain_of_thought.append({
                "step": current_query_num + 3,
                "type": "query_refinement",
                "content": f"Refined query #{current_query_num + 1}: {json.dumps(refined_query)}"
            })

            # Execute refined query
            new_results = await self.tool_agent.execute_query(refined_query)

            # Merge and deduplicate results
            filtered_results = self._merge_and_deduplicate(filtered_results, new_results)

            chain_of_thought.append({
                "step": current_query_num + 3,
                "type": "refined_results",
                "content": f"After refinement: {len(filtered_results)} total unique chunks"
            })

            current_query_num += 1

        # Final token check
        final_tokens = self._count_tokens_from_emails(filtered_results)

        if final_tokens > self.max_context_tokens:
            # Emergency filtering
            filtered_results = filtered_results[:max(1, len(filtered_results) // 2)]
            final_tokens = self._count_tokens_from_emails(filtered_results)

            chain_of_thought.append({
                "step": "emergency_filter",
                "type": "emergency_filtering",
                "content": f"Applied emergency filtering: {len(filtered_results)} chunks, {final_tokens} tokens"
            })

            if final_tokens > self.max_context_tokens:
                return {
                    "error": f"Unable to reduce context to fit within {self.max_context_tokens} token limit. Found relevant chunks but context is too large.",
                    "chain_of_thought": chain_of_thought,
                    "emails_found": len(email_results),
                    "final_emails": final_tokens
                }

        # Step 4: Generate final response with email context
        final_response = await self._generate_final_response(user_prompt, filtered_results)

        chain_of_thought.append({
            "step": "final",
            "type": "response_generation",
            "content": f"Generated response using {len(filtered_results)} chunks as context ({final_tokens} tokens)"
        })

        return {
            "response": final_response,
            "chain_of_thought": chain_of_thought,
            "context_tokens_used": final_tokens,
            "queries_executed": current_query_num,
            "emails_analyzed": final_tokens
        }

    async def _create_plan(self, user_prompt: str) -> str:
        """Create an initial plan for querying emails"""
        plan_prompt = f"""
        You are an email analysis assistant. Create a focused search strategy to find relevant information from emails to answer this user question:

        Question: {user_prompt}

        Identify:
        1. Key keywords and topics to search for
        2. Potential email senders or recipients that might be relevant
        3. Time periods that might be important
        4. The type of information needed (specific facts, summaries, discussions, etc.)

        Keep your plan concise and actionable.

        Search Strategy:
        """

        response = await self.llm_service.generate(plan_prompt, max_tokens=300)
        return response.strip()

    async def _generate_initial_query(self, user_prompt: str, plan: str) -> Dict[str, Any]:
        """Generate the initial broad query as a dict with optional filters"""
        query_prompt = f"""
        Based on this search strategy, create a search query to find relevant emails:

        User Question: {user_prompt}
        Search Strategy: {plan}

        Create a search query that will capture emails related to the user's question. Focus on the most important keywords and concepts.

        Search Query:
        """
        response = await self.llm_service.generate(query_prompt, max_tokens=100)
        query_text = response.strip()
        filters = self._extract_filters(user_prompt, plan)

        return {"text": query_text, "limit": 20, **filters}

    async def _refine_query(
        self,
        user_prompt: str,
        plan: str,
        current_results: List[Dict[str, Any]],
        iteration: int,
    ) -> Dict[str, Any]:
        """Generate a refined query based on current results"""

        # Analyze current results to inform refinement
        subjects = [r.get('subject', '') for r in current_results[:10]]
        senders = list(set([r.get('sender', '') for r in current_results[:10]]))

        refinement_prompt = f"""
        I need to refine my email search to better answer this question:

        User Question: {user_prompt}
        Original Plan: {plan}
        Iteration: {iteration}

        Current results include emails with subjects like: {', '.join(subjects[:5])}
        From senders like: {', '.join(senders[:3])}

        Create a more specific search query to find the most relevant emails for answering the user's question.
        Focus on being more precise and targeted.

        Refined Search Query:
        """

        response = await self.llm_service.generate(refinement_prompt, max_tokens=100)
        query_text = response.strip()
        filters = self._extract_filters(user_prompt, plan)
        return {"text": query_text, "limit": 20, **filters}

    def _extract_filters(self, *texts: str) -> Dict[str, Any]:
        """Extract sender and date filters from a series of texts"""
        combined = " ".join(t for t in texts if t)
        filters: Dict[str, Any] = {}

        sender_match = re.search(r"from\s+([\w.\-]+@[\w\.-]+)", combined, re.IGNORECASE)

        if sender_match:
            filters["sender"] = sender_match.group(1)

        start = None
        end = None

        exact_match = re.search(r"on\s+(\d{4}-\d{2}-\d{2})", combined)

        if exact_match:
            start = exact_match.group(1)
            end = exact_match.group(1)

        after_match = re.search(r"(?:after|since)\s+(\d{4}-\d{2}-\d{2})", combined, re.IGNORECASE)
        if after_match:
            start = after_match.group(1)

        before_match = re.search(r"before\s+(\d{4}-\d{2}-\d{2})", combined, re.IGNORECASE)
        if before_match:
            end = before_match.group(1)

        if start:
            filters["start_date"] = start
        if end:
            filters["end_date"] = end

        return filters

    def _merge_and_deduplicate(
            self,
            existing_results: List[Dict[str, Any]],
            new_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge results and remove duplicates"""

        # Create a set of existing IDs
        existing_ids = {result.get('id', '') for result in existing_results}

        # Add new results that aren't already present
        merged = existing_results.copy()
        for result in new_results:
            if result.get('id', '') not in existing_ids:
                merged.append(result)

        # Sort by relevance score (descending)
        merged.sort(key=lambda x: x.get('score', 0), reverse=True)

        return merged

    def __metadata_filter(
        self,
        results: List[Dict[str, Any]],
        user_prompt: str,
        plan: str,
    ) -> List[Dict[str, Any]]:
        """Filter results using metadata clues and token limits"""

        filters = self._extract_filters(user_prompt, plan)

        if filters.get("sender"):
            results = [r for r in results if r.get("sender", "").lower() == filters["sender"].lower()]

        start = filters.get("start_date")
        end = filters.get("end_date")
        if start or end:
            def in_range(date_str: str) -> bool:
                if not date_str:
                    return False
                if start and date_str < start:
                    return False
                if end and date_str > end:
                    return False
                return True

            results = [r for r in results if in_range(r.get("date", ""))]

        # Keep only the highest scoring results
        sorted_results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)

        # Start with top results and add until we approach token limit
        filtered = []
        running_tokens = 0
        target_tokens = int(self.max_context_tokens * 0.8)  # Use 80% of limit for safety

        for result in sorted_results:
            result_tokens = self._count_tokens_from_emails([result])
            if running_tokens + result_tokens <= target_tokens:
                filtered.append(result)
                running_tokens += result_tokens
            else:
                break

        # Ensure we have at least one result
        if not filtered and sorted_results:
            # Take the top result and truncate its content if necessary
            top_result = sorted_results[0].copy()
            content = top_result.get('content', '')
            if len(content) > 1000:  # Truncate long content
                top_result['content'] = content[:1000] + "... [truncated]"
            filtered = [top_result]

        return filtered

    def _count_tokens_from_emails(self, email_results: List[Dict[str, Any]]) -> int:
        """Count tokens from email results"""
        if not email_results:
            return 0

        content_text = "\n".join(email.get("content", "") for email in email_results)
        return len(self.encoder.encode(content_text))

    async def _generate_final_response(
            self,
            user_prompt: str,
            email_results: List[Dict[str, Any]]
    ) -> str:
        """Generate the final response using email context"""

        if not email_results:
            return "I couldn't find any relevant emails to answer your question. Please try rephrasing your query or ensure the relevant emails have been uploaded."

        # Format email context for the LLM
        email_context = self._format_email_context(email_results)

        final_prompt = f"""
        You are an intelligent email assistant. Answer the user's question based upon the Email Context..

        User Question: {user_prompt}

        Email Context ({len(email_results)} emails):
        {email_context}

        Instructions:
        1. Answer the question based on the information found in the emails
        2. If the emails don't contain enough information to fully answer the question, say so.
        3. Return response in a conversational format unless the user specifies a format or persona.

        Answer:
        """

        response = await self.llm_service.generate(final_prompt, max_tokens=1000)
        return response.strip()

    def _format_email_context(self, email_results: List[Dict[str, Any]]) -> str:
        """Format email results into readable context for the LLM"""

        formatted_emails = []

        for email in email_results[:20]:  # Limit to top 20 emails
            subject = email.get('subject', 'No Subject')
            email_text = f"""
EMAIL {subject}:
Subject: {subject}
From: {email.get('sender', 'Unknown Sender')}
To: {email.get('recipient', 'Unknown Recipient')}
Date: {email.get('date', 'Unknown Date')}
Relevance Score: {email.get('score', 0):.3f}

Content:
{email.get('content', 'No content available')[:2000]}{"..." if len(email.get('content', '')) > 2000 else ""}

---
"""
            formatted_emails.append(email_text)

        return "\n".join(formatted_emails)
