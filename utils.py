from chats import alpaca
from chats import baize
from chats import flan
from chats import llama_rlhf
from chats import glm

def get_chat_interface(model_type, batch_enabled):
    if model_type == "Alpaca":
        return alpaca.chat_batch if batch_enabled else alpaca.chat_stream
    elif model_type == "Baize":
        return baize.chat_batch if batch_enabled else baize.chat_stream
    elif model_type == "Flan":
        return flan.chat_batch if batch_enabled else flan.chat_stream
    elif model_type == "LLaMA":
        return llama_rlhf.chat_batch if batch_enabled else llama_rlhf.chat_stream    
    elif model_type == "ChatGLM":
        return glm.chat_batch if batch_enabled else glm.chat_stream    
    else:
        return None
