# RFC MCP

This is a weekend project to create a simple RAG application that stores embeddings of text documents using Qdrant and a Claude MCP.

## Steps

1. Jupyter notebook `idx_qdrant.ipynb` has the Python code to index documents into Qdrant.
2. Copy `.env.example` to `.env` and fill in the values.
3. Run the MCP server using `./run_server.sh`.
4. In another terminal, run the MCP client using `./run_client.sh`.
