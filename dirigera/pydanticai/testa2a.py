#
# Test A2A communication to the A2A agent (when run as a module with uvicorn
#
import httpx
import uuid

task_id = str(uuid.uuid4())
payload = {
    "jsonrpc": "2.0",
    "method": "tasks/send",
    "id": "req-1",
    "params": {
        "id": task_id,
        "message": {
            "role": "user",
            "parts": [{"type": "text", "text": "Hello from Agent B!"}]
        }
    }
}

with httpx.Client() as client:
    response = client.get("http://localhost:8000/.well-known/agent.json")
    print("Agent card:")
    print(response.json())

    response = client.post("http://localhost:8000", json=payload)
    print("Response:")
    print(response.json())
