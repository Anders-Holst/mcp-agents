#
# Test MCP communication to the MCP server running sse and a webcam capture.
#
import asyncio
from fastmcp import Client
from fastmcp.client.transports import SSETransport
import base64
import cv2
import numpy as np

async def main():
    # Connect via stdio to a local script
    async with Client(transport=SSETransport("http://127.0.0.1:8000/sse")) as client:
        tools = await client.list_tools()
        print("Available tools:")
        for tool in tools:
            print(tool)
        result = await client.call_tool("capture_webcam", {})

        # If the base64 string includes a data URI scheme (e.g., "data:image/png;base64,..."), remove the prefix
        firstResult = result[0]
        if firstResult.type == "image":
            print("Image received - mime type: " + firstResult.mimeType)
            base64_str = firstResult.data

            # Decode base64 into bytes
            image_data = base64.b64decode(base64_str)
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_data, np.uint8)

            # Decode image from numpy array using OpenCV
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            cv2.imshow("Webcam", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())