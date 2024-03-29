import streamlit as st
import torch
from transformers import AutoModel, AutoTokenizer,AutoModelForCausalLM
from transformers.generation import GenerationConfig
import json


from modelscope.hub.snapshot_download import snapshot_download

model_dir = snapshot_download('Xunzillm4cc/Xunzi-Qwen-Chat', 
                              cache_dir='./Xunzi-Qwen-Chat', 
                              revision='master')

# 設定頁面標題、圖示和版面
st.set_page_config(
    page_title="荀子大模型效果演示",
    page_icon=":robot:",
    layout="wide"
)
# 在Sider中加入圖片
from PIL import Image
image = Image.open('./web/荀子logonew.png')
st.sidebar.image(image, width=300)
## 設定為模型ID或本機資料夾路徑
# model_path = "/home/gpuall/ifs_data/pre_llms/Xunzi-Qwen-Chat"
# model_path = "./Xunzi-Qwen-Chat"
model_path = "/content/XunziALLM/Xunzi-Qwen-Chat/Xunzillm4cc/Xunzi-Qwen-Chat"


@st.cache_resource
def get_model():
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).cuda()
    model = model.eval()
    return tokenizer, model


# 載入Xunzi的model和tokenizer
tokenizer, model = get_model()
config = GenerationConfig.from_pretrained(
        model_path, trust_remote_code=True, resume_download=True,
    )
# 初始化歷史記錄和past key values
if "history" not in st.session_state:
    st.session_state.history = []
if "past_key_values" not in st.session_state:
    st.session_state.past_key_values = None

# 設定max_length、top_p和temperature
max_new_tokens = st.sidebar.slider("max_new_tokens", 0, 2048, 1024, step=1)
top_p = st.sidebar.slider("top_p", 0.0, 1.0, 0.8, step=0.01)
top_k = st.sidebar.slider("top_k", 0.0, 1.0, 0.0, step=0.01)
# temperature = st.sidebar.slider("temperature", 0.0, 1.0, 0.6, step=0.01)

config.max_new_tokens=max_new_tokens
config.top_p=top_p
config.top_k=top_k
# 清除歷史對話紀錄
buttonClean = st.sidebar.button("清除歷史對話紀錄", key="clean")
if buttonClean:
    st.session_state.history = []
    st.session_state.past_key_values = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    st.rerun()

# 渲染聊天歷史記錄
for i, message in enumerate(st.session_state.history):
    if message["role"] == "user":
        with st.chat_message(name="user", avatar="./web/用户头像.png"):
            st.markdown(message["content"])
    else:
        with st.chat_message(name="assistant", avatar="./web/机器人头像.png"):
            st.markdown(message["content"])

# 輸入框和輸出框
with st.chat_message(name="user", avatar="./web/用户头像.png"):
    input_placeholder = st.empty()
with st.chat_message(name="assistant", avatar="./web/机器人头像.png"):
    message_placeholder = st.empty()

# 取得使用者輸入
prompt_text = st.chat_input("請輸入您的問題")

def json2tuple(json_list):
    tuple_list = []
    for i in range(0, len(json_list) - 1, 2):
        if json_list[i]['role'] == 'user' and json_list[i+1]['role'] == 'assistant':
            tuple_list.append((json_list[i]['content'], json_list[i+1]['content']))
    return tuple_list
    
    
    

# 對於千問系列模型來說，需要對上下文記錄進行一定修改
if prompt_text:

    input_placeholder.markdown(prompt_text)
    history = st.session_state.history
    history_=json2tuple(history)
    history.append({"role":"user","content":prompt_text})
    past_key_values = st.session_state.past_key_values
    for response in model.chat_stream(
        tokenizer,
        prompt_text,
        history=history_,
        generation_config=config
    ):  
        
        message_placeholder.markdown(response)
    response=response
    history.append({"role":"assistant","content":response})
    print(history)

    # 更新歷史記錄和past key values
    st.session_state.history = history
    st.session_state.past_key_values = past_key_values
