from fastmcp import FastMCP
from mss import mss
from PIL import Image
import io   
import base64
import argparse

mcp = FastMCP("ScreenshotServer")

@mcp.tool()
def capture_screen(region: list = None) -> str:
    """
    Capture a screenshot.

    Args:
        region (list): Optional. A list of four integers [left, top, width, height] specifying the region to capture.

    Returns:
        str: Base64-encoded PNG image of the screenshot.
    """
    with mss() as sct:
        monitor = sct.monitors[1]  # Default to the first monitor
        if region and len(region) == 4:
            monitor = {
                "left": region[0],
                "top": region[1],
                "width": region[2],
                "height": region[3]
            }
        screenshot = sct.grab(monitor)
        img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return "data:image/png;base64," + encoded_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Screenshot MCP Server')
    parser.add_argument('--host', default="127.0.0.1", help='Host to bind to')
    parser.add_argument('--port', default=8000, type=int, help='Port to bind to')
    parser.add_argument('--transport', default="stdio", help='Transport to use (stdio, sse or http)')
    args = parser.parse_args()
    if args.transport != "stdio":
        mcp.run(transport=args.transport, host=args.host, port=args.port)
    else:
        mcp.run()