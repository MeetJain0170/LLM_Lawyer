# Legal LLM Frontend

A ChatGPT-like frontend interface for the Legal LLM Assistant.

## Features

- 🎨 Modern ChatGPT-like UI design
- 💬 Real-time chat interface
- 📝 Chat history management
- ⚡ Fast and responsive
- 🔄 Auto-resizing input
- 📱 Mobile responsive

## Setup

1. Make sure the backend API server is running on `http://localhost:5000`
2. Open `index.html` in your web browser, or use a local server:

### Using Python's HTTP Server:
```bash
cd frontend
python3 -m http.server 8000
```
Then open `http://localhost:8000` in your browser.

### Using Node.js (if you have it):
```bash
cd frontend
npx http-server -p 8000
```

## Usage

1. Start the backend API server (see backend README)
2. Open the frontend in your browser
3. Type your legal questions and get AI-powered responses!

## API Endpoints

The frontend expects the following API endpoints:

- `GET /health` - Health check
- `POST /chat` - Send a message and get a response

Request format:
```json
{
  "message": "Your question here",
  "max_new_tokens": 200,
  "temperature": 0.55,
  "top_k": 20
}
```

Response format:
```json
{
  "response": "AI response text",
  "inference_time_seconds": 0.123
}
```

