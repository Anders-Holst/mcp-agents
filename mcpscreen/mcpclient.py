#
# Test MCP communication to the MCP server running sse
#
import asyncio
from fastmcp import Client
from fastmcp.client.transports import SSETransport
import base64

async def main():
    # Connect via stdio to a local script
    async with Client(transport=SSETransport("http://127.0.0.1:8000/sse")) as client:
        tools = await client.list_tools()
        print("Available tools:")
        for tool in tools:
            print(tool)
        result = await client.call_tool("capture_screen", {})

        # If the base64 string includes a data URI scheme (e.g., "data:image/png;base64,..."), remove the prefix
        text = result[0].text
        if "data:image" in text:
            result = text.split(",")[1]

        # Decode the base64 string into binary data
        image_data = base64.b64decode(result)

        # Specify the output file name and path
        output_file = "output_image.png"

        # Write the binary data to the file in binary write mode
        with open(output_file, "wb") as file:
            file.write(image_data)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())