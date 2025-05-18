#
# Test A2A communication to the A2A agent (when run as a module with uvicorn
#
import asyncio
from fastmcp import Client
from fastmcp.client.transports import SSETransport

async def main():
    # Connect via stdio to a local script
    async with Client(transport=SSETransport("http://127.0.0.1:8000/sse")) as client:
        tools = await client.list_tools()
        print("Available tools:")
        for tool in tools:
            print(tool)
        result = await client.call_tool("get_lights", {})
        print(f"Available Lights: {result}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
