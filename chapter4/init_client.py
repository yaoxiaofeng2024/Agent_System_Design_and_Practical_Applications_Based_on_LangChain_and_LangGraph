from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

#deepseek_api = os.environ['DEEPSEEK_API']
#deepseek_url = os.environ['DEEPSEEK_URL']
#deepseek_model = os.environ['DEEPSEEK_MODEL']
deepseek_api = 'sk-b444e6b29b344981bbd848db8fe7fe0b'
deepseek_url = 'https://api.deepseek.com'
deepseek_model = 'deepseek-v4-pro'

def init_llm(temperature):
    deepseek_client = ChatOpenAI(api_key=deepseek_api,
                                 base_url=deepseek_url,
                                 model=deepseek_model,
                                 temperature=temperature)
    return deepseek_client


