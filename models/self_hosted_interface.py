# from ollama import Client
#
#
# def instantiate_ollama_client(host, port):
#     try:
#         ollama_client = Client(host=f"http://{host}:{port}", timeout=120.0)
#         return ollama_client
#     except Exception as e:
#         print("Error instantiating Ollama client: ", str(e))
#         return None
#
# llm_models = {
#     "mistral_latest": 'mistral:latest',
#     "gemma_latest": 'gemma3:latest'
# }


embedding_models = {
    "minilm": "all-MiniLM-L6-v2"
}