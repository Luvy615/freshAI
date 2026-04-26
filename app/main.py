# -*- coding: utf-8 -*-
"""
生鲜电商对话式推荐系统 - Streamlit Web界面
"""

import os
import sys
import streamlit as st
from pathlib import Path

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

from demo.fresh_food_recommender import (
    FreshFoodRecommender, llm_generate, clean_str
)
from demo.fresh_food_agent import FreshFoodAgent


st.set_page_config(
    page_title="生小鲜 - 智能生鲜推荐",
    page_icon="🥬",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.markdown("""
<style>
    * {
        font-family: "SimSun", "宋体", serif !important;
    }
    
    .stTextInput > div > div > input {
        font-family: "SimSun", "宋体", serif !important;
        border-radius: 24px;
        padding: 14px 20px;
        font-size: 1rem;
    }
    
    .stButton > button {
        font-family: "SimSun", "宋体", serif !important;
        border-radius: 24px;
        padding: 12px 36px;
        font-size: 1rem;
    }
    
    .main-header {
        text-align: center;
        padding: 25px 0;
        border-bottom: 2px solid #4CAF50;
        margin-bottom: 30px;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        color: #2E7D32;
        margin: 0;
        letter-spacing: 10px;
        font-weight: normal;
    }
    
    .main-header p {
        color: #888;
        font-size: 0.95rem;
        margin-top: 10px;
    }
    
    .chat-box {
        background: #FAFAFA;
        border-radius: 16px;
        padding: 25px;
        min-height: 380px;
        max-height: 480px;
        overflow-y: auto;
        border: 1px solid #E8E8E8;
    }
    
    .chat-placeholder {
        text-align: center;
        color: #AAA;
        padding: 80px 20px;
    }
    
    .chat-placeholder h3 {
        color: #4CAF50;
        font-weight: normal;
    }
    
    .user-msg {
        background: linear-gradient(135deg, #E8F5E9, #C8E6C9);
        padding: 14px 20px;
        border-radius: 18px 18px 4px 18px;
        margin: 12px 0;
        max-width: 75%;
        margin-left: auto;
        color: #1B5E20;
        font-size: 0.95rem;
    }
    
    .user-msg-label {
        font-size: 0.7rem;
        color: #4CAF50;
        margin-bottom: 4px;
    }
    
    .ai-msg {
        background: #FFF;
        padding: 14px 20px;
        border-radius: 18px 18px 18px 4px;
        margin: 12px 0;
        max-width: 75%;
        color: #333;
        border: 1px solid #E8E8E8;
        font-size: 0.95rem;
    }
    
    .ai-msg-label {
        font-size: 0.7rem;
        color: #4CAF50;
        margin-bottom: 6px;
    }
    
    .product-list {
        margin-top: 15px;
    }
    
    .product-item {
        background: linear-gradient(135deg, #FFF8E1, #FFECB3);
        border-left: 4px solid #FF9800;
        padding: 14px 18px;
        margin: 10px 0;
        border-radius: 8px;
    }
    
    .product-item-name {
        font-size: 0.95rem;
        font-weight: bold;
        color: #E65100;
        margin-bottom: 8px;
    }
    
    .product-item-price {
        font-size: 1.2rem;
        color: #F44336;
        font-weight: bold;
    }
    
    .product-item-info {
        color: #777;
        font-size: 0.8rem;
        margin-top: 6px;
    }
    
    .input-area {
        margin-top: 20px;
        display: flex;
        gap: 12px;
        align-items: center;
    }
    
    .input-wrapper {
        flex: 1;
    }
    
    .sidebar-title {
        text-align: center;
        padding: 15px 0;
        border-bottom: 1px solid #E8E8E8;
    }
    
    .sidebar-title h2 {
        color: #2E7D32;
        font-weight: normal;
        font-size: 1.2rem;
    }
    
    .category-grid {
        padding: 15px 10px;
    }
    
    .footer {
        text-align: center;
        padding: 20px;
        color: #CCC;
        font-size: 0.75rem;
        margin-top: 20px;
    }
    
    .loading-spinner {
        text-align: center;
        color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_agent():
    return FreshFoodAgent()


def init_session():
    if "agent" not in st.session_state:
        st.session_state.agent = init_agent()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "input_key" not in st.session_state:
        st.session_state.input_key = 0


def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>生小鲜</h1>
        <p>智能生鲜商品推荐</p>
    </div>
    """, unsafe_allow_html=True)


def render_chat():
    if not st.session_state.messages:
        st.markdown("""
        <div class="chat-box">
            <div class="chat-placeholder">
                <h3>您好，我是生小鲜</h3>
                <p style="margin-top: 15px;">请告诉我您想购买的生鲜商品</p>
                <p style="margin-top: 8px; color: #4CAF50;">例如：苹果、草莓、排骨、鱼虾...</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown('<div class="chat-box">', unsafe_allow_html=True)
    
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="user-msg">
                <div class="user-msg-label">您</div>
                {msg["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            content = msg["content"]
            if msg.get("type") == "products":
                st.markdown(f"""
                <div class="ai-msg">
                    <div class="ai-msg-label">生小鲜</div>
                    {content}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="ai-msg">
                    <div class="ai-msg-label">生小鲜</div>
                    {content}
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


def format_products_html(products):
    if not products:
        return "<p>暂无推荐商品</p>"
    
    items = []
    for i, p in enumerate(products[:6], 1):
        name = clean_str(p.get("productName", p.get("product_name", "")))
        price = p.get("price", 0)
        sales = p.get("sales", 0)
        taste = clean_str(p.get("taste", "")) if p.get("taste") else "暂无"
        
        items.append(f"""
        <div class="product-item">
            <div class="product-item-name">{i}. {name}</div>
            <div>
                <span class="product-item-price">{price:.2f}元</span>
                <span class="product-item-info">销量 {sales} | 口感 {taste}</span>
            </div>
        </div>
        """)
    
    return f'<div class="product-list">{"".join(items)}</div>'


def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-title">
            <h2>🥬 商品分类</h2>
        </div>
        """, unsafe_allow_html=True)
        
        categories = ["水果", "蔬菜", "肉类", "海鲜", "禽蛋", "豆制品", "奶制品"]
        
        for cat in categories:
            if st.button(f"📦 {cat}", key=f"cat_{cat}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": f"我想买{cat}"})
                process_search(f"我想买{cat}")
        
        st.markdown("<hr style='margin: 20px 0; border: none; border-top: 1px solid #E8E8E8;'>", unsafe_allow_html=True)
        
        if st.button("🗑️ 清除对话", key="clear", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.agent:
                st.session_state.agent.clear_history()
            st.rerun()
        
        st.markdown("<hr style='margin: 20px 0; border: none; border-top: 1px solid #E8E8E8;'>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; color: #999; font-size: 0.8rem;">
            <p>已连接: Neo4j | Milvus</p>
        </div>
        """, unsafe_allow_html=True)


def process_search(query):
    agent = st.session_state.get("agent")
    if not agent:
        agent = init_agent()
        st.session_state.agent = agent
    
    with st.spinner("正在搜索商品..."):
        try:
            products = agent.get_products(query)
            
            if products:
                products_html = format_products_html(products)
                response = f"为您找到 {len(products)} 款商品：<br>{products_html}<br><p style='color:#888;margin-top:15px;'>请问还有什么需要帮助的？</p>"
            else:
                result = agent.run(query)
                response = f"<p>{result}</p>"
            
            st.session_state.messages.append({"role": "user", "content": query})
            st.session_state.messages.append({"role": "assistant", "content": response, "type": "products"})
        except Exception as e:
            error_msg = f"<p>抱歉，搜索遇到问题: {str(e)}</p>"
            st.session_state.messages.append({"role": "user", "content": query})
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    st.rerun()


def main():
    init_session()
    render_sidebar()
    render_header()
    render_chat()
    
    with st.container():
        col1, col2 = st.columns([6, 1])
        with col1:
            user_input = st.text_input(
                "",
                placeholder="请输入您想购买的生鲜商品...",
                key=f"input_{st.session_state.input_key}",
                label_visibility="collapsed"
            )
        with col2:
            search_clicked = st.button("搜索", type="primary", use_container_width=True)
        
        if search_clicked and user_input:
            process_search(user_input)
            st.session_state.input_key += 1
    
    st.markdown("""
    <div class="footer">
        <p>Powered by Neo4j + Milvus + Qwen LLM</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()