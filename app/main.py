# -*- coding: utf-8 -*-
"""
生鲜电商对话式推荐系统 - Streamlit Web界面
支持：
- 文本对话搜索商品
- 上传图片搜索商品
- 对话形式展示推荐商品
"""

import os
import sys
import io
import streamlit as st
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

from demo.fresh_food_recommender import (
    FreshFoodRecommender, llm_generate, clean_str
)
from demo.fresh_food_agent import FreshFoodAgent


# ==================== 页面配置 ====================
st.set_page_config(
    page_title="生小鲜 - 智能生鲜推荐",
    page_icon="🥬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================== CSS样式 ====================
def load_custom_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .product-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
    }
    
    .product-name {
        font-size: 1.1rem;
        font-weight: bold;
        color: #1b5e20;
    }
    
    .product-price {
        font-size: 1.3rem;
        color: #e53935;
        font-weight: bold;
    }
    
    .product-info {
        color: #616161;
        font-size: 0.9rem;
    }
    
    .chat-message {
        padding: 10px 15px;
        border-radius: 15px;
        margin: 5px 0;
    }
    
    .user-message {
        background-color: #e8f5e9;
        border-bottom-right-radius: 0;
    }
    
    .assistant-message {
        background-color: #f5f5f5;
        border-bottom-left-radius: 0;
    }
    
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
    }
    
    .stButton>button:hover {
        background-color: #388E3C;
    }
    
    .upload-hint {
        color: #9e9e9e;
        font-size: 0.8rem;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)


# ==================== 初始化 ====================
@st.cache_resource
def init_agent():
    return FreshFoodAgent()


def init_session_state():
    if "agent" not in st.session_state:
        st.session_state.agent = init_agent()
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "你好！我是生小鲜，一个智能生鲜推荐助手。🥬\n\n你可以：\n- 告诉我你想买什么生鲜（比如苹果、草莓、排骨）\n- 上传图片搜索相似商品\n\n有什么可以帮到你的？", "type": "text"}
        ]
    
    if "current_image" not in st.session_state:
        st.session_state.current_image = None


# ==================== 侧边栏 ====================
def render_sidebar():
    with st.sidebar:
        st.title("🥬 生小鲜")
        st.markdown("---")
        
        st.subheader("📂 商品分类")
        categories = ["水果", "蔬菜", "肉禽", "海鲜", "豆制品", "禽蛋", "奶制品"]
        for cat in categories:
            if st.button(f"📦 {cat}", key=f"cat_{cat}"):
                st.session_state.messages.append({
                    "role": "user", 
                    "content": f"我想买{cat}", 
                    "type": "text"
                })
                process_user_input(f"我想买{cat}")
        
        st.markdown("---")
        
        st.subheader("🗑️ 操作")
        if st.button("🗑️ 清除对话", key="clear_chat"):
            st.session_state.messages = [
                {"role": "assistant", "content": "对话已清除。有什么可以帮到你的？", "type": "text"}
            ]
            if st.session_state.agent:
                st.session_state.agent.clear_history()
            st.rerun()
        
        if st.button("🖼️ 清除图片", key="clear_image"):
            st.session_state.current_image = None
            if st.session_state.agent:
                st.session_state.agent.clear_image()
            st.rerun()
        
        st.markdown("---")
        
        st.subheader("ℹ️ 系统信息")
        st.markdown(f"""
        - ** Neo4j**: 已连接 ✓
        - ** Milvus**: 已连接 ✓
        - ** DeepSeek**: 已连接 ✓
        """)


# ==================== 对话区域 ====================
def render_chat():
    st.markdown('<p class="main-header">🥬 生小鲜 - 智能生鲜推荐</p>', unsafe_allow_html=True)
    
    # 显示对话历史
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🥬"):
                if msg.get("type") == "products":
                    render_products(msg["content"])
                else:
                    st.markdown(msg["content"])


def render_products(products_data):
    """渲染商品卡片"""
    if isinstance(products_data, list):
        products = products_data
    else:
        products = []
    
    if not products:
        st.markdown("暂无推荐商品")
        return
    
    cols = st.columns(2)
    for i, p in enumerate(products[:6]):
        with cols[i % 2]:
            with st.container():
                st.markdown(f"""
                <div class="product-card">
                    <div class="product-name">{i+1}. {clean_str(p.get('productName', p.get('product_name', '')))}</div>
                    <div class="product-price">💰 {p.get('price', 0)}元</div>
                    <div class="product-info">
                        📦 销量: {p.get('sales', 0)}件<br>
                        🏪 {clean_str(p.get('shop', '未知店铺'))}<br>
                        🍯 口感: {clean_str(p.get('taste', '暂无'))}
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ==================== 输入区域 ====================
def render_input():
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.chat_input("请描述你想买的生鲜商品...")
    
    with col2:
        uploaded_file = st.file_uploader(
            "", 
            type=['jpg', 'jpeg', 'png'],
            key="image_uploader",
            help="上传图片搜索商品"
        )
    
    # 处理图片上传
    if uploaded_file is not None:
        st.session_state.current_image = uploaded_file
        st.markdown(f"✅ 图片已上传: {uploaded_file.name}")
    
    if user_input:
        process_user_input(user_input)


def process_user_input(user_input: str):
    """处理用户输入"""
    agent = st.session_state.get("agent")
    if not agent:
        agent = init_agent()
        st.session_state.agent = agent
    
    # 用户消息
    st.session_state.messages.append({"role": "user", "content": user_input, "type": "text"})
    
    # 获取推荐
    with st.spinner("正在为你推荐商品..."):
        try:
            products = agent.get_products(user_input, st.session_state.current_image)
            
            if products:
                # 推荐商品消息
                response = f"为你找到了 {len(products)} 款合适的生鲜商品：\n\n"
                response += format_product_list(products)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response, 
                    "type": "products"
                })
            else:
                # 尝试纯对话生成
                result = agent.run(user_input, st.session_state.current_image)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": result, 
                    "type": "text"
                })
                
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"抱歉，搜索遇到一些问题: {str(e)}。请尝试其他描述~", 
                "type": "text"
            })
    
    # 保持图片状态
    if not st.session_state.get("clear_image_flag"):
        st.session_state.current_image = st.session_state.get("current_image")
    
    st.rerun()


def format_product_list(products: list) -> str:
    """格式化商品列表"""
    lines = []
    for i, p in enumerate(products[:6], 1):
        name = clean_str(p.get("productName", p.get("product_name", "")))
        price = p.get("price", 0)
        sales = p.get("sales", 0)
        taste = clean_str(p.get("taste", ""))
        shop = clean_str(p.get("shop", ""))
        
        lines.append(f"**{i}. {name}**")
        lines.append(f"   💰 {price}元 | 📦 {sales}件 | 🏪 {shop}")
        if taste:
            lines.append(f"   🍯 口感: {taste}")
        lines.append("")
    
    return "\n".join(lines)


# ==================== 主函数 ====================
def main():
    load_custom_css()
    init_session_state()
    render_sidebar()
    render_chat()
    
    if len(st.session_state.messages) <= 1:
        render_input()
    else:
        render_input()


if __name__ == "__main__":
    main()