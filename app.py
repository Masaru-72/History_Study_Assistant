from fastapi import FastAPI, Request, Form, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import json
import base64
from query_data import query_rag_stream
from langchain.memory import ConversationBufferMemory
from typing import Optional, AsyncIterable

# Initialize the FastAPI app
app = FastAPI()
templates = Jinja2Templates(directory="templates")
session_logs = [] # Note: This is a simple in-memory log, not suitable for production.

# Allow CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Serves the main index.html file at the root URL.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws/ask")
async def stream_answer(websocket: WebSocket):
    await websocket.accept()
    # Create a new memory instance for each connection to isolate conversations.
    memory = ConversationBufferMemory(memory_key="history", return_messages=False)
    try:
        while True:
            json_str = await websocket.receive_text()
            data = json.loads(json_str)
            query = data.get("query")
            image_b64 = data.get("image")  # Image is a base64 data URL

            if not query and not image_b64:
                continue

            image_bytes = None
            if image_b64:
                try:
                    header, encoded = image_b64.split(",", 1)
                    image_bytes = base64.b64decode(encoded)
                except Exception as e:
                    print(f"Error decoding base64 image: {e}")
                    await websocket.send_text("[ERROR] Invalid image data.")
                    continue

            # Pass the session-specific memory object to the RAG function.
            full_response = ""
            lang = data.get("lang", "en")
            speed = float(data.get("speed", 1.0))
            async for token in query_rag_stream(query_text=query, memory=memory, image_bytes=image_bytes, speed=speed, lang=lang):
                await websocket.send_text(token)
                full_response += token
            
            # Send a special End-Of-Stream message.
            await websocket.send_text("[END_OF_STREAM]")

            # Log the conversation after the full response is sent
            if query and full_response:
                session_logs.append({
                    "user": query,
                    "bot": full_response.strip(),
                    "timestamp": datetime.now().isoformat()
                })


    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"An error occurred in WebSocket: {e}")
        await websocket.close()

@app.get("/api/logs", response_class=JSONResponse)
def get_and_save_logs():
    if session_logs:
        try:
            with open("chat_log.json", "r", encoding="utf-8") as file:
                existing_logs = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_logs = []
        
        existing_logs.extend(session_logs)
        
        with open("chat_log.json", "w", encoding="utf-8") as file:
            json.dump(existing_logs, file, ensure_ascii=False, indent=2)
            
        session_logs.clear()

    try:
        with open("chat_log.json", "r", encoding="utf-8") as f:
            all_logs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_logs = []
    
    return {"logs": all_logs}

@app.get("/logs", response_class=HTMLResponse)
def show_logs_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})