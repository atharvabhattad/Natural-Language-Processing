# Dual-Agent Query Answering System

This project employs a dual-agent architecture to answer queries by combining real-time data from the Tavily API with summarization capabilities from the Ethernia text generation model. The system leverages Langchain and Langgraph to construct and coordinate the agents.

## Workflow Overview

### Imports and API Keys
- Imports modules for web search, text processing, and natural language processing.
- Sets API keys for Tavily (web search) and Hugging Face (text generation).

### Embedding Wrapper
- Implements a custom wrapper for the SentenceTransformer model.
- Converts text into numerical embeddings for similarity searches.

### Research Agent
- Uses the Tavily API to perform web searches.
- Processes and splits the returned text into manageable chunks.
- Builds a FAISS vector store to efficiently retrieve relevant information.

### Answer Drafting Agent
- Queries the FAISS vector store to retrieve pertinent documents based on the query.
- Concatenates document texts to form a comprehensive context.
- Uses a Hugging Face text generation model to produce a coherent final answer.

### Workflow Setup
- Creates a workflow graph using Langgraph.
- Connects agents sequentially: the research agent runs first and passes its results to the answer drafting agent, which then generates the final answer.

## References
- [Tavily Documentation](https://docs.tavily.com/welcome)
- [Langgraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Langchain Tutorials](https://python.langchain.com/v0.2/docs/tutorials/)

## Future Enhancements
- Implement more advanced models for text generation.
- Expand the workflow with additional agents (e.g., for fact-checking and deeper research).
- Experiment with various prompt strategies, including chain-of-thought approaches.
