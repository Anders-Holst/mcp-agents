from fastmcp import FastMCP, Image
from ollama import chat
import argparse
import cv2

mcp = FastMCP("WebcamServer")
# Keep kamera "running" for faster image capture
webcam = cv2.VideoCapture(0)

@mcp.tool()
def capture_webcam_image() -> Image:
    """
    Capture a webcam image.

    Returns:
        Image: Base64-encoded PNG image of the webcam image.
    """
    ret, frame = webcam.read()
    if not ret:
        return "Failed to capture image"
    # Encode the image to PNG format in memory
    success, encoded_image = cv2.imencode('.png', frame, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not success:
        return "Failed to encode image as PNG"
    return Image(data=encoded_image.tobytes(), format="png")

@mcp.tool()
def analyze_webcam_image(prompt: str) -> str:
    """
    Analyze a freshly captured webcam image. 
    Args:
        prompt (str): The prompt used for the analysis of the image.

    Returns:
        str: Analysis of the webcam image.
    """
    ret, frame = webcam.read()
    if not ret:
        return "Failed to capture image"
    
    # Encode the image to PNG format in memory
    success, encoded_image = cv2.imencode('.png', frame, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not success:
        return "Failed to encode image as PNG"
    
    # Convert to bytes
    image_bytes = encoded_image.tobytes()

    res = chat('gemma3:4b', messages=[
        {'role': 'user', 'images': [image_bytes], 'content': prompt},
    ], stream=False)

    return res['message']['content']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Webcam MCP Server')
    parser.add_argument('--host', default="127.0.0.1", help='Host to bind to')
    parser.add_argument('--port', default=8000, type=int, help='Port to bind to')
    parser.add_argument('--transport', default="stdio", help='Transport to use (stdio, sse or http)')
    args = parser.parse_args()
    if args.transport != "stdio":
        mcp.run(transport=args.transport, host=args.host, port=args.port)
    else:
        mcp.run()