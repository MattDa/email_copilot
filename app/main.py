# app/main.py
import os
import asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import uvicorn

from services.email_processor import EmailProcessor
from services.chroma_service import ChromaService
from services.llm_service import LLMService
from agents.plan_execute_agent import PlanExecuteAgent

app = FastAPI(title="Email Copilot", version="1.0.0")

# Initialize services
chroma_service = ChromaService()
llm_service = LLMService()
email_processor = EmailProcessor(chroma_service)
plan_execute_agent = PlanExecuteAgent(chroma_service, llm_service)

templates = Jinja2Templates(directory="app/templates")


@app.on_event("startup")
async def startup_event():
    """Initialize services and process existing emails"""
    await chroma_service.initialize()
    await email_processor.process_existing_emails("/app/data/emails")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload-email")
async def upload_email(files: list[UploadFile] = File(...)):
    """Upload and process multiple .msg files"""
    results = []
    errors = []

    for file in files:
        if not file.filename.endswith('.msg'):
            errors.append({"filename": file.filename, "error": "Only .msg files are supported"})
            continue

        try:
            # Save uploaded file
            file_path = f"/app/data/uploads/{file.filename}"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

            # Process the email
            await email_processor.process_single_email(file_path)

            results.append({"filename": file.filename, "status": "success"})
        except Exception as e:
            errors.append({"filename": file.filename, "error": str(e)})

    if errors and not results:
        # If all files failed, return a 500 error
        raise HTTPException(status_code=500, detail={"message": "All files failed to process", "errors": errors})

    return {
        "message": f"Successfully processed {len(results)} out of {len(results) + len(errors)} files",
        "successful": results,
        "failed": errors
    }

@app.post("/query")
async def query_emails(query: dict):
    """Process user query and return response"""
    user_prompt = query.get("prompt", "")
    if not user_prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    try:
        response = await plan_execute_agent.execute(user_prompt)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/debug/parse-msg")
async def debug_parse_msg(file: UploadFile = File(...)):
    """Debug endpoint to test MSG parsing"""
    if not file.filename.endswith('.msg'):
        raise HTTPException(status_code=400, detail="Only .msg files are supported")

    try:
        # Save uploaded file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.msg') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        # Parse the file
        from utils.msg_parser import MSGParser
        parser = MSGParser()
        parsed_data = parser.parse_msg_file(tmp_file_path)

        # Clean up temp file
        os.unlink(tmp_file_path)

        if parsed_data:
            return {
                "success": True,
                "subject": parsed_data.get('subject'),
                "sender": parsed_data.get('sender'),
                "body_length": len(parsed_data.get('body', '')),
                "body_preview": parsed_data.get('body', '')[:300]
            }
        else:
            return {"success": False, "error": "Failed to parse MSG file"}

    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)

