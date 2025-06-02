# R1_RAGReadFromLocal
This repo allows the deepseek R1 model to read local files to allow for Retrieval-Augmented Generation response. 

It does this by front loading the chat history with the specified materials.

This is useful in generating data sets for model training - we can use the more specific responses.

Resources used - heavily relied on:
https://github.com/coleam00/ottomator-agents/tree/main/r1-distill-rag
https://www.youtube.com/watch?v=uWDocIoiaXE



py -m venv venv

Linux/MacOS:
source venv/bin/activate

Windows:
.\venv\Scripts\activate

pip install -r dependencies.txt

If we need a larger local context window for the LLM :
# Create Deepseek model with 8k context - recommended for reasoning
ollama create deepseek-r1:7b-8k -f ollama_models/Deepseek-r1-7b-8k

# Create Qwen model with 8k context - recommended for conversation
ollama create qwen2.5:14b-instruct-8k -f ollama_models/Qwen-14b-Instruct-8k

# On MacOS you might need to use -from instead of -f

# Create Deepseek model with 8k context - recommended for reasoning
ollama create deepseek-r1:7b-8k -from ollama_models/Deepseek-r1-7b-8k

# Create Qwen model with 8k context - recommended for conversation
ollama create qwen2.5:14b-instruct-8k -from ollama_models/Qwen-14b-Instruct-8k
