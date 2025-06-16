# app/utils/msg_parser.py
import extract_msg
from typing import Dict, Any, Optional
from datetime import datetime
import hashlib
import re


class MSGParser:
    def parse_msg_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Parse .msg file and extract relevant information"""
        try:
            msg = extract_msg.Message(file_path)

            # Get body content - try multiple methods with better cleaning
            body_content = self._extract_body_content(msg)

            # Clean and validate content
            body_content = self._clean_content(body_content)

            # Extract email data
            email_data = {
                'subject': self._clean_content(msg.subject or 'No Subject'),
                'sender': self._clean_content(msg.sender or 'Unknown Sender'),
                'recipient': ', '.join(msg.to) if msg.to else 'Unknown Recipient',
                'cc': ', '.join(msg.cc) if msg.cc else '',
                'bcc': ', '.join(msg.bcc) if msg.bcc else '',
                'date': str(msg.date) if msg.date else 'Unknown Date',
                'body': body_content,
                'message_id': self._generate_message_id(msg),
                'file_path': file_path
            }

            # Debug print to see what we extracted
            print(f"Parsed email: Subject='{email_data['subject'][:50]}...', Body length={len(body_content)}")
            if body_content:
                print(f"Body preview: '{body_content[:100]}...'")
            else:
                print("WARNING: No body content extracted!")

            msg.close()
            return email_data

        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None

    def _extract_body_content(self, msg) -> str:
        """Extract body content using multiple methods"""
        body_content = ""

        # Method 1: Try plain text body first
        if hasattr(msg, 'body') and msg.body:
            body_content = str(msg.body)
            print(f"Extracted body using .body: {len(body_content)} chars")

        # Method 2: Try HTML body if plain text is not available
        elif hasattr(msg, 'htmlBody') and msg.htmlBody:
            html_content = str(msg.htmlBody)
            # Strip HTML tags
            body_content = self._strip_html(html_content)
            print(f"Extracted body using .htmlBody: {len(body_content)} chars")

        # Method 3: Try RTF body
        elif hasattr(msg, 'rtfBody') and msg.rtfBody:
            body_content = str(msg.rtfBody)
            print(f"Extracted body using .rtfBody: {len(body_content)} chars")

        # Method 4: Check if there's a text version in attachments or properties
        else:
            # Sometimes the body is in different properties
            try:
                # Try accessing raw properties
                if hasattr(msg, '_getStringStream'):
                    # This is a more advanced approach for stubborn MSG files
                    pass
                print("No body content found in standard properties")
            except Exception as e:
                print(f"Error trying alternative body extraction: {e}")

        return body_content

    def _strip_html(self, html_content: str) -> str:
        """Strip HTML tags and decode entities"""
        import html

        # Remove HTML tags
        clean_text = re.sub(r'<[^>]+>', '', html_content)

        # Decode HTML entities
        clean_text = html.unescape(clean_text)

        # Clean up whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text)

        return clean_text.strip()

    def _clean_content(self, content: str) -> str:
        """Clean content by removing null bytes and other problematic characters"""
        if not content:
            return ""

        # Convert to string if it isn't already
        content = str(content)

        # Remove null bytes and other control characters
        content = content.replace('\x00', '')  # Remove null bytes
        content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', content)  # Remove other control chars

        # Clean up excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()

        return content

    def _generate_message_id(self, msg) -> str:
        """Generate a unique message ID"""
        # Use message properties to create unique ID
        subject = str(msg.subject or '')
        sender = str(msg.sender or '')
        date = str(msg.date or '')
        id_string = f"{subject}{sender}{date}"
        return hashlib.md5(id_string.encode()).hexdigest()