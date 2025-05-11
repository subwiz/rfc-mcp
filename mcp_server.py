import asyncio
import json
import logging
import os
from typing import Dict, List, Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import qdrant_client
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Qdrant MCP Server")

# Add CORS middleware for browser requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, set this to your Claude Desktop origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "text_embeddings")
MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
DEFAULT_LIMIT = int(os.getenv("DEFAULT_LIMIT", "5"))

# Global variables for clients
model = None
client = None

# MCP Protocol Models
class SearchRequest(BaseModel):
    query: str
    limit: int = Field(default=5, ge=1, le=20)

class SearchResult(BaseModel):
    file_name: str
    text: str
    similarity_score: float

class MCPMessage(BaseModel):
    type: str
    data: Dict[str, Any]

@app.on_event("startup")
async def startup_event():
    """Initialize Qdrant client and embedding model on startup"""
    global model, client

    # Initialize embedding model
    logger.info(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # Initialize Qdrant client
    logger.info(f"Connecting to Qdrant at {QDRANT_URL}")
    client = qdrant_client.QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY if QDRANT_API_KEY else None
    )

    # Verify collection exists
    try:
        collections = client.get_collections().collections
        if not any(collection.name == COLLECTION_NAME for collection in collections):
            logger.error(f"Collection {COLLECTION_NAME} does not exist in Qdrant")
            raise ValueError(f"Collection {COLLECTION_NAME} does not exist")
        logger.info(f"Connected to collection: {COLLECTION_NAME}")
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {str(e)}")
        raise

@app.get("/")
async def root():
    """Root endpoint - basic info about the MCP server"""
    return {
        "name": "Qdrant Semantic Search MCP Server",
        "description": "Connects Claude Desktop to Qdrant vector database via MCP",
        "status": "online"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.websocket("/mcp")
async def mcp_websocket(websocket: WebSocket):
    """WebSocket endpoint implementing the Model Context Protocol for Claude Desktop"""
    await websocket.accept()
    logger.info("WebSocket connection established")

    try:
        # Send MCP hello message per protocol specification
        await websocket.send_json({
            "type": "hello",
            "data": {
                "name": "Qdrant Document Search",
                "capabilities": ["search"],
                "description": "Search through document embeddings stored in Qdrant"
            }
        })

        # Main communication loop
        async for message in websocket.iter_json():
            try:
                # Parse the message
                msg_type = message.get("type")
                msg_data = message.get("data", {})
                logger.info(f"Received message type: {msg_type}")

                # Handle message based on type
                if msg_type == "ping":
                    await websocket.send_json({"type": "pong", "data": {}})

                elif msg_type == "search":
                    # Extract search parameters
                    query = msg_data.get("query")
                    limit = min(msg_data.get("limit", DEFAULT_LIMIT), 20)  # Cap at 20 for performance

                    if not query:
                        await websocket.send_json({
                            "type": "error",
                            "data": {"message": "Search query cannot be empty"}
                        })
                        continue

                    # Perform search
                    logger.info(f"Searching for: '{query}' with limit {limit}")
                    results = await search_documents(query, limit)

                    # Send search results
                    await websocket.send_json({
                        "type": "search_results",
                        "data": {
                            "results": [result.dict() for result in results],
                            "query": query,
                            "total": len(results)
                        }
                    })

                elif msg_type == "error":
                    logger.error(f"Received error from client: {msg_data}")

                else:
                    logger.warning(f"Unknown message type: {msg_type}")
                    await websocket.send_json({
                        "type": "error",
                        "data": {"message": f"Unknown message type: {msg_type}"}
                    })

            except json.JSONDecodeError:
                logger.error("Failed to parse WebSocket message as JSON")
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": "Invalid JSON message"}
                })

            except Exception as e:
                logger.exception("Error processing message")
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": f"Error processing message: {str(e)}"}
                })

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")

    except Exception as e:
        logger.exception("WebSocket error")
        try:
            await websocket.close()
        except:
            pass

async def search_documents(query: str, limit: int = 5) -> List[SearchResult]:
    """
    Search for documents in Qdrant based on semantic similarity

    This function is compatible with qdrant-client version 1.14.2
    """
    if not model or not client:
        raise RuntimeError("Search services not initialized")

    # Generate embedding for the query
    query_embedding = model.encode(query).tolist()

    # For qdrant-client 1.14.2, we should use the search method with these parameters
    try:
        logger.info(f"Searching with query vector length: {len(query_embedding)}")
        search_results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=limit
        )

        # Format results
        results = []
        for result in search_results:
            results.append(SearchResult(
                file_name=result.payload.get("file_name", "unknown"),
                text=result.payload.get("text", ""),
                similarity_score=float(result.score)
            ))

        return results

    except Exception as e:
        logger.error(f"Search error: {str(e)}", exc_info=True)
        # More detailed error information to help debug
        logger.error(f"Parameters: collection_name={COLLECTION_NAME}, vector_size={len(query_embedding)}, limit={limit}")
        raise RuntimeError(f"Search failed: {str(e)}")

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8765"))
    logger.info(f"Starting MCP server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
