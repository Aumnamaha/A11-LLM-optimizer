import sys
import os
import time
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from engine.pipeline import A11Pipeline

app = FastAPI(title="A11-LLM-Optimizer Local Inference Server")

pipeline_instance: Optional[A11Pipeline] = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "Qwen3.8-Flash-Next-UD-Q2_K_XL"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False

@app.on_event("startup")
def startup_event():
    global pipeline_instance
    model_dir = "/home/knk/Projects/A11-LLM-optimizer/models/Qwen3-30B-A3B"
    print("\n[Server Boot] Initializing A11 Engine Pipeline...")
    pipeline_instance = A11Pipeline(model_dir, vram_budget_gb=4.0, ram_budget_gb=16.0)

@app.on_event("shutdown")
def shutdown_event():
    global pipeline_instance
    if pipeline_instance:
        print("\n[Server Shutdown] Cleaning up engine pipeline resources...")
        pipeline_instance.shutdown()

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "Qwen3.8-Flash-Next-UD-Q2_K_XL",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "A11-LLM-Optimizer"
            }
        ]
    }

@app.post("/v1/chat/completions")
def create_chat_completion(request: ChatCompletionRequest):
    global pipeline_instance
    if not pipeline_instance:
        raise HTTPException(status_code=500, detail="Engine pipeline not initialized.")
    
    user_prompt = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_prompt = msg.content
            break
            
    if not user_prompt:
        user_prompt = "Hello!"

    # Handle Server-Sent Events (SSE) Streaming
    if request.stream:
        def event_generator():
            created_time = int(time.time())
            for token in pipeline_instance.stream_inference(user_prompt):
                chunk = {
                    "id": f"chatcmpl-a11-{created_time}",
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": token},
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # Handle Standard Non-Streaming JSON Response
    start_t = time.time()
    tokens = list(pipeline_instance.stream_inference(user_prompt))
    full_text = "".join(tokens)
    elapsed = time.time() - start_t

    return {
        "id": f"chatcmpl-a11-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": full_text
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(user_prompt.split()),
            "completion_tokens": len(tokens),
            "total_tokens": len(user_prompt.split()) + len(tokens)
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("engine.server:app", host="127.0.0.1", port=8080, reload=False)
