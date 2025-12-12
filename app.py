"""
Universal Quiz Solver API - FastAPI Server
Main entry point for the quiz solving service.
"""

import os
import asyncio
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import solver core
from solver.solver_core import QuizSolver

app = FastAPI(
    title="Universal Quiz Solver API",
    description="An autonomous Quiz-Solver Agent that handles JavaScript-rendered quiz pages",
    version="1.0.0"
)

# Environment variable for secret validation
SOLVER_SECRET = os.getenv("SOLVER_SECRET", "default_secret")


class QuizRequest(BaseModel):
    """Request model for quiz solving endpoint."""
    url: str
    secret: str
    timeout: Optional[int] = 180  # 3 minutes default


class QuizResponse(BaseModel):
    """Response model for quiz solving endpoint."""
    status: str
    message: Optional[str] = None
    task_id: Optional[str] = None


# Store for tracking background tasks
task_results = {}


async def solve_quiz_background(task_id: str, url: str, timeout: int):
    """Background task to solve a quiz."""
    try:
        logger.info(f"[{task_id}] Starting quiz solving for URL: {url}")
        solver = QuizSolver(timeout=timeout)
        result = await solver.solve(url)
        task_results[task_id] = {
            "status": "completed",
            "result": result
        }
        logger.info(f"[{task_id}] Quiz solving completed successfully")
    except Exception as e:
        logger.error(f"[{task_id}] Quiz solving failed: {str(e)}")
        task_results[task_id] = {
            "status": "failed",
            "error": str(e)
        }


@app.post("/solve", response_model=QuizResponse)
async def solve_quiz(request: QuizRequest, background_tasks: BackgroundTasks):
    """
    Main endpoint to solve a quiz.
    
    - Validates JSON body
    - Validates secret using SOLVER_SECRET env var
    - Returns 400 if JSON invalid
    - Returns 403 if secret mismatch
    - Returns 200 { "status": "accepted" } if OK
    - Launches solving pipeline asynchronously
    """
    # Validate secret
    if request.secret != SOLVER_SECRET:
        raise HTTPException(
            status_code=403,
            detail="Invalid secret provided"
        )
    
    # Validate URL
    if not request.url or not request.url.startswith(("http://", "https://")):
        raise HTTPException(
            status_code=400,
            detail="Invalid URL format"
        )
    
    # Generate task ID
    import uuid
    task_id = str(uuid.uuid4())[:8]
    
    # Initialize task tracking
    task_results[task_id] = {"status": "processing"}
    
    # Launch background task
    background_tasks.add_task(
        solve_quiz_background,
        task_id,
        request.url,
        request.timeout
    )
    
    return QuizResponse(
        status="accepted",
        message="Quiz solving started",
        task_id=task_id
    )


@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of a quiz solving task."""
    if task_id not in task_results:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return task_results[task_id]


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
    """Handle Pydantic validation errors."""
    return JSONResponse(
        status_code=400,
        content={"detail": "Invalid JSON body", "errors": exc.errors()}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
