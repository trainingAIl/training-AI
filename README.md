# Custom Personal AI Model & FastAPI Service

A custom-trained character-level LSTM neural network built with PyTorch and deployed via FastAPI. This model answers personal questions based on a custom dataset without using external LLM APIs.

## Project Structure
- `api.py`: FastAPI application serving the `/ask` endpoint for model inference.
- `my_ai_model.pt`: Trained PyTorch model weights (100 epochs).
- `requirements.txt`: Required Python dependencies (`fastapi`, `uvicorn`, `torch`, `pydantic`).

## API Usage
Once deployed, access the interactive Swagger UI documentation at `/docs` and send a `POST` request to `/ask`:

```json
{
  "question": "What is my height?"
}
