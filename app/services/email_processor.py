# app/services/email_processor.py
import os
import asyncio
from typing import List
from pathlib import Path

from utils.msg_parser import MSGParser
from services.chroma_service import ChromaService


class EmailProcessor:
    def __init__(self, chroma_service: ChromaService):
        self.chroma_service = chroma_service
        self.msg_parser = MSGParser()

    async def process_existing_emails(self, email_folder: str):
        """Process all existing .msg files in the folder"""
        email_path = Path(email_folder)
        if not email_path.exists():
            print(f"Email folder {email_folder} does not exist")
            return

        msg_files = list(email_path.glob("**/*.msg"))
        print(f"Found {len(msg_files)} .msg files to process")

        for msg_file in msg_files:
            try:
                await self.process_single_email(str(msg_file))
            except Exception as e:
                print(f"Error processing {msg_file}: {e}")

    async def process_single_email(self, file_path: str):
        """Process a single .msg file"""
        print(f"Processing email file: {file_path}")
        email_data = self.msg_parser.parse_msg_file(file_path)
        if email_data:
            print(
                f"Parsed email data: Subject='{email_data.get('subject')}', Body length={len(email_data.get('body', ''))}")
            await self.chroma_service.add_email(email_data)
            print(f"Successfully processed: {email_data.get('subject', 'No Subject')}")

        else:
            print(f"Failed to parse email file: {file_path}")