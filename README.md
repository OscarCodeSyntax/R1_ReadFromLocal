# R1_RAGReadFromLocal
This repo allows the deepseek R1 model to read local files to allow for Retrieval-Augmented Generation response. 

It does this by front loading the chat history with the specified materials.

This is useful in generating data sets for model training - we can use the more specific responses.

Resources used - heavily relied on:
https://github.com/coleam00/ottomator-agents/tree/main/r1-distill-rag
https://www.youtube.com/watch?v=uWDocIoiaXE


*Initalize virtual environment*
py -m venv venv

*Linux/MacOS:*
source venv/bin/activate

*Windows:*
.\venv\Scripts\activate

*install deps*
pip install -r dependencies.txt

ollama list in virtual enviroment to see model names available

If we need a larger local context window for the LLM :
# Create Deepseek model with 8k context - recommended for reasoning
ollama create deepseek-r1:7b-8k -f ollama_models/Deepseek-r1-7b-8k

# Create Qwen model with 8k context - recommended for conversation
ollama create qwen2.5:14b-instruct-8k -f ollama_models/Qwen-14b-Instruct-8k


# On MacOS you might need to use -from instead of -f

# Create Deepseek model with 8k context - recommended for reasoning
ollama create deepseek-r1:7b-8k -from ollama_models/Deepseek-r1-7b-8k

# Create Qwen model with 8k context - recommended for conversation
ollama create qwen2.5:14b-instruct-8k -from ollama_models/qwen2.5:14b-instruct



**How It Works**
Document Ingestion (ingest_pdfs.py):

**Loads PDFs from the data directory**
- Splits documents into chunks of 1000 characters with 200 character overlap
- Creates embeddings using sentence-transformers/all-mpnet-base-v2
- Stores vectors in a Chroma database
- RAG System (r1_smolagent_rag.py):

**Uses two LLMs: one for reasoning and one for tool calling**

Retrieves relevant document chunks based on user queries
Generates responses using the retrieved context
Provides a Gradio web interface for interaction

**Model Selection**
*HuggingFace Models
- Recommended for cloud-based inference
- Requires API token for better rate limits
- Supports a wide range of models
- Better for production use with stable API

*Ollama Models*
- Recommended for local inference
- No API token required
- Runs entirely on your machine
- Better for development and testing
- Supports custom model configurations
- Lower latency but requires more system resources

**Notes**
The vector store is persisted in the chroma_db directory
Default chunk size is 1000 characters with 200 character overlap
Embeddings are generated using the all-mpnet-base-v2 model
The system uses a maximum of 3 relevant chunks for context



# Create Deepseek model with 8k context - recommended for reasoning
ollama create deepseek-r1:7b-8k -f ollama_models/Deepseek-r1-7b-8k

# Create Qwen model with 8k context - recommended for conversation
ollama create qwen2.5:14b-instruct-8k -f ollama_models/Qwen-14b-Instruct-8k

# On MacOS you might need to use -from instead of -f

# Create Deepseek model with 8k context - recommended for reasoning
ollama create deepseek-r1:7b-8k -from ollama_models/Deepseek-r1-7b-8k

# Create Qwen model with 8k context - recommended for conversation
ollama create qwen2.5:14b-instruct-8k -from ollama_models/Qwen-14b-Instruct-8k