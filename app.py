from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from scripts.main import stream_chat_response, get_chat_metadata


app = FastAPI(
    title="Cortex AI API",
    description="Backend API for Cortex AI coding assistant"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# REQUEST MODELS
# =========================================================
class ChatRequest(BaseModel):
    message: str
    thread_id: str


class MetadataRequest(BaseModel):
    thread_id: str


# =========================================================
# ROOT
# =========================================================
@app.get("/")
async def root():
    return {
        "message": "Cortex AI API is running"
    }


# =========================================================
# STREAM CHAT
# =========================================================
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):

    def event_generator():
        try:
            for chunk in stream_chat_response(
                user_message=request.message,
                thread_id=request.thread_id
            ):
                if hasattr(chunk[0], "content"):
                    token = chunk[0].content

                    if token:
                        yield token

        except Exception as e:
            yield f"\n[ERROR]: {str(e)}"

    return StreamingResponse(
        event_generator(),
        media_type="text/plain"
    )


# =========================================================
# FETCH METADATA
# =========================================================
@app.post("/chat/metadata")
async def chat_metadata(request: MetadataRequest):
    try:
        metadata = get_chat_metadata(request.thread_id)
        return metadata

    except Exception as e:
        return {
            "error": str(e)
        }