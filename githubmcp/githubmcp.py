#
# MCP Example accessing GitHub issues.
# Note: this will not list all issues - just first "page".
#

from fastmcp import FastMCP
import requests, argparse

mcp = FastMCP("GitHub Issues MCP")

@mcp.tool()
def list_issues(owner: str, repo: str, state: str = "open") -> list[dict]:
    """
    List issues from a public GitHub repository.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    params = {"state": state}
    headers = {"Accept": "application/vnd.github.v3+json"}
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    return response.json()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GitHub Issues MCP Server')
    parser.add_argument('--host', default="127.0.0.1", help='Host to bind to')
    parser.add_argument('--port', default=8000, type=int, help='Port to bind to')
    parser.add_argument('--transport', default="stdio", help='Transport to use (stdio, sse or http)')
    args = parser.parse_args()
    if args.transport != "stdio":
        mcp.run(transport=args.transport, host=args.host, port=args.port)
    else:
        mcp.run()
