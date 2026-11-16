from chatbot import answer_question
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="MIT 6.006 RAG Chatbot API",
    description="API for an educational RAG chatbot",
    version="1.0.0"
)


class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        description="Question asked by the user"
    )


class ChatResponse(BaseModel):
    answer: str
    sources: list = []


@app.get("/")
def home():
    return {
        "message": "MIT 6.006 RAG API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):

    try:
        # We will connect your existing RAG function here
        answer = answer_question(
            query=request.question,
            context=request.history
        )
        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
