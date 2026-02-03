from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from src.agent import Graph
import json
import os

app = FastAPI(title="Medical Reference Chatbot")

# Store active sessions: session_id -> Graph instance
sessions: dict[str, Graph] = {}


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    
    # Create a new Graph instance for this session
    sessions[session_id] = Graph()
    graph = sessions[session_id]
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_message = message_data.get("message", "")
            
            if not user_message:
                continue
            
            # Stream response tokens back to client
            async for token in graph.stream(user_message):
                await websocket.send_text(token)
            
            # Signal end of response
            await websocket.send_text("[END]")
    
    except WebSocketDisconnect:
        # Clean up session on disconnect
        if session_id in sessions:
            del sessions[session_id]
    
    except Exception as e:
        # Send error to client and clean up
        await websocket.send_text(f"Error: {str(e)}")
        await websocket.send_text("[END]")
        if session_id in sessions:
            del sessions[session_id]


@app.get("/health")
async def health_check():
    return {"status": "healthy", "active_sessions": len(sessions)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 30080)))