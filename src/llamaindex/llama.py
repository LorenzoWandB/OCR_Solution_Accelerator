import os
from getpass import getpass
from dotenv import load_dotenv

load_dotenv()


def get_llama_api_key() -> str:
    if "LLAMA_CLOUD_API_KEY" not in os.environ:
        api_key = getpass("Enter your Llama Cloud API Key: ")
        os.environ["LLAMA_CLOUD_API_KEY"] = api_key
        return api_key
    
    return os.environ["LLAMA_CLOUD_API_KEY"]


def check_setup() -> bool:
    try:
        from llama_cloud_services import LlamaExtract
        
        api_key = os.environ.get("LLAMA_CLOUD_API_KEY")
        if not api_key:
            print("❌ LLAMA_CLOUD_API_KEY not found in environment")
            print("   Please set it in your .env file or run get_llama_api_key()")
            return False
        
        # Try to initialize client
        client = LlamaExtract(api_key=api_key)
        print("✅ LlamaExtract is properly configured")
        return True
        
    except ImportError:
        print("❌ llama-cloud-services not installed")
        print("   Run: pip install llama-cloud-services")
        return False
    except Exception as e:
        print(f"❌ Error checking LlamaExtract setup: {e}")
        return False


if __name__ == "__main__":
    print("🔧 Checking LlamaExtract Setup...")
    check_setup()
