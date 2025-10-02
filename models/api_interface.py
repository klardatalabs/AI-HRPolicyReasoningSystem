from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# gemini_api_key=os.getenv("GEMINI_API_KEY", "")

def instantiate_openai_client(key: str):
    try:
        openai_client = OpenAI(
            api_key=key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        return openai_client
    except Exception as e:
        print("Error instantiating OpenAI client: ", str(e))
        return None

