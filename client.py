import asyncio
import json
import logging
import sys
import os
from typing import Dict, List, Any, Optional, Union
import websockets
import requests
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("mcp-client")

# Configuration from environment variables
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-7-sonnet-20250219")
QDRANT_MCP_URL = os.getenv("QDRANT_MCP_URL", "ws://localhost:8765/mcp")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))

class MCPClient:
    """Client for connecting to an MCP server and integrating with Claude API"""

    def __init__(self, mcp_url: str, anthropic_api_key: str, model: str):
        """
        Initialize the MCP client

        Args:
            mcp_url: WebSocket URL for the MCP server
            anthropic_api_key: API key for Claude
            model: Claude model to use
        """
        self.mcp_url = mcp_url
        self.anthropic_client = Anthropic(api_key=anthropic_api_key)
        self.model = model
        self.ws = None
        self.server_info = None
        self.connected = False

    async def connect(self) -> bool:
        """
        Connect to the MCP server

        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info(f"Connecting to MCP server at {self.mcp_url}")
            self.ws = await websockets.connect(self.mcp_url)

            # Wait for hello message
            hello_msg = await self.ws.recv()
            hello_data = json.loads(hello_msg)

            if hello_data.get("type") != "hello":
                logger.error(f"Expected 'hello' message, got: {hello_data.get('type')}")
                await self.disconnect()
                return False

            self.server_info = hello_data.get("data", {})
            logger.info(f"Connected to MCP server: {self.server_info.get('name')}")
            logger.info(f"Server capabilities: {self.server_info.get('capabilities', [])}")

            self.connected = True
            return True

        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {str(e)}")
            return False

    async def disconnect(self) -> None:
        """Close the WebSocket connection"""
        if self.ws:
            await self.ws.close()
            self.ws = None
            self.connected = False
            logger.info("Disconnected from MCP server")

    async def ping(self) -> bool:
        """
        Send a ping message to the server

        Returns:
            True if pong received, False otherwise
        """
        if not self.connected or not self.ws:
            logger.error("Cannot ping: not connected")
            return False

        try:
            await self.ws.send(json.dumps({"type": "ping", "data": {}}))
            response = await self.ws.recv()
            response_data = json.loads(response)

            return response_data.get("type") == "pong"
        except Exception as e:
            logger.error(f"Ping failed: {str(e)}")
            return False

    async def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents using the MCP server

        Args:
            query: The search query
            limit: Maximum number of results to return

        Returns:
            List of search results
        """
        if not self.connected or not self.ws:
            logger.error("Cannot search: not connected")
            return []

        try:
            logger.info(f"Searching for: '{query}' with limit {limit}")

            # Send search request
            await self.ws.send(json.dumps({
                "type": "search",
                "data": {
                    "query": query,
                    "limit": limit
                }
            }))

            # Wait for response
            response = await self.ws.recv()
            response_data = json.loads(response)

            # Check if we got search results
            if response_data.get("type") != "search_results":
                logger.error(f"Expected 'search_results', got: {response_data.get('type')}")
                return []

            results = response_data.get("data", {}).get("results", [])
            logger.info(f"Received {len(results)} search results")
            return results

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []

    async def query_claude_with_context(
        self,
        user_query: str,
        search_query: Optional[str] = None,
        max_results: int = 3
    ) -> str:
        """
        Query Claude with context from document search

        Args:
            user_query: The user's question
            search_query: Custom search query (if None, will use user_query)
            max_results: Maximum number of search results to include in context

        Returns:
            Claude's response
        """
        # Use user query as search query if none provided
        search_query = search_query or user_query

        # Search for relevant documents
        search_results = await self.search(search_query, limit=max_results)

        if not search_results:
            logger.warning("No search results found, querying Claude without context")
            return await self.query_claude(user_query)

        # Format search results as context
        context = "\n\n".join([
            f"Document: {result.get('file_name')}\n"
            f"Relevance: {result.get('similarity_score'):.4f}\n"
            f"Content: {result.get('text')}"
            for result in search_results
        ])

        # Create system prompt with search results
        system_prompt = (
            "You are Claude, an AI assistant with access to a document database. "
            "Below are documents retrieved based on the user's query. "
            "Use this information to answer the user's question as accurately as possible. "
            "If the retrieved documents don't contain relevant information, acknowledge this "
            "and answer based on your general knowledge.\n\n"
            "RETRIEVED DOCUMENTS:\n"
            f"{context}"
        )

        # Query Claude with the context
        try:
            logger.info(f"Querying Claude with {len(search_results)} documents as context")

            response = self.anthropic_client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=system_prompt,
                messages=[{"role": "user", "content": user_query}]
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"Claude API request failed: {str(e)}")
            return f"Error querying Claude: {str(e)}"

    async def query_claude(self, user_query: str) -> str:
        """
        Query Claude without additional context

        Args:
            user_query: The user's question

        Returns:
            Claude's response
        """
        try:
            logger.info(f"Querying Claude without additional context")

            response = self.anthropic_client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": user_query}]
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"Claude API request failed: {str(e)}")
            return f"Error querying Claude: {str(e)}"

async def interactive_session():
    """Run an interactive session with the MCP client"""
    if not ANTHROPIC_API_KEY:
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        return

    client = MCPClient(
        mcp_url=QDRANT_MCP_URL,
        anthropic_api_key=ANTHROPIC_API_KEY,
        model=CLAUDE_MODEL
    )

    # Connect to MCP server
    connected = await client.connect()
    if not connected:
        print("Failed to connect to MCP server. Exiting.")
        return

    print(f"Connected to {client.server_info.get('name', 'MCP Server')}")
    print("Type 'exit' or 'quit' to end the session.")
    print("Type 'search: your query' to search documents directly.")
    print("--------------------------------------------------------")

    try:
        while True:
            # Get user input
            user_input = input("\nYou: ").strip()

            # Check for exit command
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting...")
                break

            # Check for direct search command
            if user_input.lower().startswith("search:"):
                search_query = user_input[7:].strip()
                if search_query:
                    print(f"Searching for: {search_query}")
                    results = await client.search(search_query)
                    if results:
                        print(f"\nFound {len(results)} results:")
                        for i, result in enumerate(results, 1):
                            print(f"\n--- Result {i} ---")
                            print(f"File: {result.get('file_name')}")
                            print(f"Score: {result.get('similarity_score'):.4f}")
                            print(f"Preview: {result.get('text')[:200]}...")
                    else:
                        print("No results found.")
                    continue

            # Process regular query through Claude
            print("Querying Claude with document context...")
            response = await client.query_claude_with_context(user_input)

            print("\nClaude:", response)

    except KeyboardInterrupt:
        print("\nSession interrupted. Exiting...")

    finally:
        # Clean up
        await client.disconnect()

if __name__ == "__main__":
    # Run the interactive session
    asyncio.run(interactive_session())
