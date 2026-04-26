# -*- coding: utf-8 -*-
"""
生鲜电商推荐系统核心模块
- Neo4j知识图谱客户端
- Milvus向量数据库客户端  
- FreshFoodRecommender推荐引擎
"""

import os
import sys
import io
import json
import re
from typing import List, Dict, Any, Optional
from datetime import datetime

import pandas as pd
import numpy as np

# Neo4j
from neo4j import GraphDatabase

# Milvus
from pymilvus import connections, Collection, DataType

# LLM
from openai import OpenAI

# 编码处理
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ==================== 配置 ====================
# 千问大模型配置
QWEN_API_KEY = "sk-1a1a1e9193034df5b8b61e601d927573"
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# Neo4j Aura配置
NEO4J_URI = "neo4j+s://9acccd6e.databases.neo4j.io"
NEO4J_USER = "9acccd6e"
NEO4J_PASSWORD = "FBPVbuhBV8LewZXyGbIfKIMucDx4V0gjDrwMaNYfvh4"

# Milvus Cloud配置
MILVUS_URI = "https://in03-c690e46590549f3.serverless.gcp-us-west1.cloud.zilliz.com"
MILVUS_TOKEN = "b78e04209431b99f6ea295cd538761f97b4b2101fbba8ce281816a1454a176e4404a4a332b1a82f33a741f3ed0cba38f9379b0f8"
COLLECTION_NAME = "fresh_product_embeddings"

# CLIP模型路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "向量数据库构建", "clip_model_cache", 
                    "openai-mirror", "clip-vit-base-patch32")


def clean_str(val):
    if val is None:
        return ""
    if isinstance(val, str):
        return ''.join(c for c in val if not (0xD800 <= ord(c) <= 0xDFFF) and ord(c) >= 0x20)
    return val


class Neo4jClient:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    def close(self):
        if self.driver:
            self.driver.close()
    
    def test_connection(self) -> bool:
        try:
            with self.driver.session() as session:
                result = session.run("MATCH (n) RETURN n LIMIT 1")
                result.consume()
            return True
        except Exception as e:
            print(f"Neo4j连接测试失败: {e}")
            return False
    
    def search_by_product_type(self, product_type: str, limit: int = 10) -> List[Dict]:
        with self.driver.session() as session:
            result = session.run("""
                MATCH (p:商品)-[:属于]->(pt:商品主体)
                WHERE pt.名称 = $product_type
                OPTIONAL MATCH (p)-[:具有口感]->(t:口感特征)
                OPTIONAL MATCH (p)-[:销售]->(shop:店铺)
                RETURN p.SKU AS SKU, p.产品名称 AS productName, p.价格 AS price,
                       p.销量 AS sales, p.付款人数 AS buyers,
                       pt.名称 AS productType,
                       COLLECT(DISTINCT t.名称) AS tastes,
                       shop.名称 AS shop, shop.地理位置 AS region
                LIMIT $limit
            """, product_type=product_type, limit=limit)
            return [dict(record) for record in result]
    
    def search_by_product_type_and_taste(self, product_type: str, taste: str, limit: int = 10) -> List[Dict]:
        with self.driver.session() as session:
            result = session.run("""
                MATCH (p:商品)-[:属于]->(pt:商品主体)
                WHERE pt.名称 = $product_type
                MATCH (p)-[:具有口感]->(t:口感特征)
                WHERE t.名称 CONTAINS $taste
                OPTIONAL MATCH (p)-[:销售]->(shop:店铺)
                RETURN p.SKU AS SKU, p.产品名称 AS productName, p.价格 AS price,
                       p.销量 AS sales, p.付款人数 AS buyers,
                       pt.名称 AS productType,
                       COLLECT(DISTINCT t.名称) AS tastes,
                       shop.名称 AS shop
                LIMIT $limit
            """, product_type=product_type, taste=taste, limit=limit)
            return [dict(record) for record in result]
    
    def search_by_category(self, category: str, limit: int = 20) -> List[Dict]:
        with self.driver.session() as session:
            result = session.run("""
                MATCH (p:商品)-[:属于]->(pt:商品主体)
                MATCH (pt:商品主体)-[:属于]->(cat:商品类别)
                WHERE cat.名称 = $category
                OPTIONAL MATCH (p)-[:具有口感]->(t:口感特征)
                OPTIONAL MATCH (p)-[:销售]->(shop:店铺)
                RETURN p.SKU AS SKU, p.产品名称 AS productName, p.价格 AS price,
                       p.销量 AS sales, p.付款人数 AS buyers,
                       pt.名称 AS productType,
                       COLLECT(DISTINCT t.名称) AS tastes,
                       shop.名称 AS shop
                ORDER BY p.销量 DESC
                LIMIT $limit
            """, category=category, limit=limit)
            return [dict(record) for record in result]
    
    def search_by_keyword(self, keyword: str, limit: int = 10) -> List[Dict]:
        with self.driver.session() as session:
            result = session.run("""
                MATCH (p:商品)
                WHERE p.产品名称 CONTAINS $keyword
                OPTIONAL MATCH (p)-[:属于]->(pt:商品主体)
                OPTIONAL MATCH (p)-[:具有口感]->(t:口感特征)
                OPTIONAL MATCH (p)-[:销售]->(shop:店铺)
                RETURN p.SKU AS SKU, p.产品名称 AS productName, p.价格 AS price,
                       p.销量 AS sales, p.付款人数 AS buyers,
                       pt.名称 AS productType,
                       COLLECT(DISTINCT t.名称) AS tastes,
                       shop.名称 AS shop
                ORDER BY p.销量 DESC
                LIMIT $limit
            """, keyword=keyword, limit=limit)
            return [dict(record) for record in result]


class MilvusClient:
    def __init__(self):
        connections.connect(alias="default", uri=MILVUS_URI, token=MILVUS_TOKEN, secure=True)
        self.collection = Collection(name=COLLECTION_NAME)
        self.collection.load()
        self._load_clip_model()
    
    def _load_clip_model(self):
        try:
            from modelscope import AutoModel, AutoProcessor
            import torch
            self.model = AutoModel.from_pretrained(MODEL_DIR)
            self.processor = AutoProcessor.from_pretrained(MODEL_DIR)
            self.model.eval()
            self._torch = torch
            print("[MilvusClient] CLIP模型加载完成")
        except Exception as e:
            print(f"[MilvusClient] CLIP模型加载失败: {e}")
            self.model = None
    
    def create_text_embedding(self, text: str) -> np.ndarray:
        if self.model is None:
            return np.zeros(512)
        with self._torch.no_grad():
            inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
            outputs = self.model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            embedding = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state[:, 0, :]
            embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
            return embedding[0].cpu().numpy()
    
    def search_by_text(self, query_text: str, top_k: int = 10) -> List[Dict]:
        if self.collection.num_entities == 0:
            return []
        query_embedding = self.create_text_embedding(query_text)
        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["SKU", "product_name", "price", "sales", "buyers", "shop", "region", "category", "product_type", "taste", "scene"]
        )
        return self._parse_results(results)
    
    def _search_by_image(self, image_path: str, top_k: int = 10) -> List[Dict]:
        if self.collection.num_entities == 0 or self.model is None:
            return []
        from PIL import Image
        img = Image.open(image_path).convert('RGB')
        with self._torch.no_grad():
            inputs = self.processor(images=img, return_tensors="pt")
            outputs = self.model.get_image_features(**inputs)
            embedding = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state[:, 0, :]
            embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
            image_embedding = embedding[0].cpu().numpy()
        results = self.collection.search(
            data=[image_embedding.tolist()],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["SKU", "product_name", "price", "sales", "buyers", "shop", "region", "category", "product_type", "taste", "scene"]
        )
        return self._parse_results(results)
    
    def search_by_text_with_filter(self, query_text: str, product_type: str = "", taste: str = "", top_k: int = 10) -> List[Dict]:
        candidates = self.search_by_text(query_text, top_k=top_k*2)
        if not product_type and not taste:
            return candidates[:top_k]
        filtered = []
        for c in candidates:
            pt_match = True
            if product_type:
                pt = c.get('product_type', c.get('productType', ''))
                pt_match = product_type in pt or pt in product_type
            if pt_match:
                filtered.append(c)
        if not filtered and product_type:
            filtered = self._get_products_by_type(product_type, limit=top_k)
        return filtered[:top_k]
    
    def _get_products_by_type(self, product_type: str, limit: int = 20) -> List[Dict]:
        if self.collection.num_entities == 0:
            return []
        try:
            results = self.collection.query(
                expr=f"product_type == '{product_type}'",
                limit=limit,
                output_fields=["SKU", "product_name", "price", "sales", "buyers", "shop", "region", "category", "product_type", "taste", "scene"]
            )
            return results
        except:
            return []
    
    def _parse_results(self, results) -> List[Dict]:
        if not results or not results[0]:
            return []
        parsed = []
        for hit in results[0]:
            entity = hit.entity
            parsed.append({
                "SKU": entity.get("SKU"),
                "product_name": entity.get("product_name"),
                "productName": entity.get("product_name"),
                "price": entity.get("price"),
                "sales": entity.get("sales"),
                "buyers": entity.get("buyers"),
                "shop": entity.get("shop"),
                "region": entity.get("region"),
                "category": entity.get("category"),
                "product_type": entity.get("product_type"),
                "productType": entity.get("product_type"),
                "taste": entity.get("taste"),
                "scene": entity.get("scene"),
                "similarity": hit.distance
            })
        return parsed
    
    def close(self):
        connections.disconnect("default")


class FreshFoodRecommender:
    def __init__(self):
        self._kg_client = None
        self._kg_online = None
        self.vector_client = None
    
    def _get_kg_client(self):
        if self._kg_online is None:
            try:
                client = Neo4jClient()
                if client.test_connection():
                    self._kg_client = client
                    self._kg_online = True
                    print("[FreshFoodRecommender] Neo4j连接成功")
                else:
                    self._kg_online = False
            except Exception as e:
                print(f"[FreshFoodRecommender] Neo4j连接失败: {e}")
                self._kg_online = False
        return self._kg_client if self._kg_online else None
    
    def _get_vector_client(self):
        if self.vector_client is None:
            try:
                self.vector_client = MilvusClient()
                print("[FreshFoodRecommender] Milvus连接成功")
            except Exception as e:
                print(f"[FreshFoodRecommender] Milvus连接失败: {e}")
                self.vector_client = None
        return self.vector_client
    
    def search_by_text_with_filter(self, query_text: str, product_type: str = "", taste: str = "", top_k: int = 10) -> List[Dict]:
        vc = self._get_vector_client()
        if vc is None:
            return []
        return vc.search_by_text_with_filter(query_text, product_type, taste, top_k)
    
    def vector_search(self, query: str, limit: int = 10) -> List[Dict]:
        vc = self._get_vector_client()
        if vc is None:
            return []
        return vc.search_by_text(query, top_k=limit)
    
    def _search_by_image(self, image_path: str, limit: int = 10) -> List[Dict]:
        vc = self._get_vector_client()
        if vc is None:
            return []
        return vc._search_by_image(image_path, top_k=limit)
    
    def kg_search(self, product_type: str = None, category: str = None, keyword: str = None, taste: str = None, limit: int = 10) -> List[Dict]:
        kg = self._get_kg_client()
        if kg is None:
            return []
        if product_type and taste:
            return kg.search_by_product_type_and_taste(product_type, taste, limit)
        elif product_type:
            return kg.search_by_product_type(product_type, limit)
        elif category:
            return kg.search_by_category(category, limit)
        elif keyword:
            return kg.search_by_keyword(keyword, limit)
        return []
    
    def merge_results(self, kg_results: List[Dict], vector_results: List[Dict], limit: int = 10) -> List[Dict]:
        all_products = {}
        for r in kg_results:
            sku = r.get("SKU") or r.get("sku")
            if sku and sku not in all_products:
                all_products[sku] = {"source": "KG", **r, "score": 1.0}
        for r in vector_results:
            sku = r.get("SKU") or r.get("sku")
            if sku:
                sim = r.get("similarity", r.get("similarity", 0))
                if sku in all_products:
                    all_products[sku]["source"] = "KG+Vec"
                    all_products[sku]["score"] += sim
                else:
                    all_products[sku] = {"source": "Vec", **r, "score": sim}
        products = sorted(all_products.values(), key=lambda x: x.get("score", 0), reverse=True)[:limit]
        return products
    
    def close(self):
        if self._kg_client:
            self._kg_client.close()
        if self.vector_client:
            self.vector_client.close()


def llm_generate(prompt: str, history: List[Dict] = None) -> str:
    client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)
    messages = [
        {"role": "system", "content": "你是一个生鲜电商推荐助手，根据用户需求推荐合适的生鲜商品。回复要简洁友好，推荐商品时说明推荐理由。"}
    ]
    if history:
        for h in history:
            messages.append({"role": h.get("role", "user"), "content": h.get("content", "")})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(model="qwen3.5-omni-plus", messages=messages, temperature=0.7)
    content = response.choices[0].message.content
    return clean_str(content)


def llm_intent_recognition(user_input: str) -> Dict[str, Any]:
    client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)
    prompt = f"""请分析用户的生鲜商品查询意图，并提取关键信息。用户输入：{user_input}。请以JSON格式返回：{{"intent": "search_products", "product_type": "商品主体类型", "tastes": ["口感偏好"], "category": "水果/蔬菜/肉禽/蛋奶/海鲜/水产/豆制品", "scene": "适用场景", "price_range": "便宜/适中/昂贵", "other_attrs": ["其他要求"]}}。只返回JSON。"""
    response = client.chat.completions.create(model="qwen3.5-omni-plus", messages=[{"role": "user", "content": prompt}], temperature=0.3)
    content = clean_str(response.choices[0].message.content)
    try:
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            return json.loads(json_match.group())
    except:
        pass
    return {"intent": "search_products", "product_type": "", "tastes": [], "category": "", "scene": "", "price_range": "", "other_attrs": []}


def call_qwen_vision(image_path: str, text: str) -> str:
    import base64
    client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)
    with open(image_path, 'rb') as f:
        image_base64 = base64.b64encode(f.read()).decode('utf-8')
    response = client.chat.completions.create(
        model='qwen-vl-flash',
        messages=[{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image_base64}'}},
            {'type': 'text', 'text': text}
        ]}],
        max_tokens=500
    )
    return clean_str(response.choices[0].message.content)


def extract_entities_from_image(image_path: str) -> Dict[str, Any]:
    prompt = """请分析这张生鲜商品图片，提取以下实体信息：请以JSON格式返回：{{"product_type": "商品主体类型", "tastes": ["口感偏好"], "category": "水果/蔬菜/肉禽/蛋奶/海鲜/水产/豆制品/其他", "scene": "适用场景", "price_range": "便宜/适中/昂贵", "other_attrs": ["其他特征"], "description": "图片中商品的详细描述"}}。只返回JSON。"""
    description = call_qwen_vision(image_path, prompt)
    try:
        json_match = re.search(r'\{[\s\S]*\}', description)
        if json_match:
            return json.loads(json_match.group())
    except:
        pass
    return {"product_type": "", "tastes": [], "category": "", "scene": "", "price_range": "", "other_attrs": [], "description": ""}


if __name__ == "__main__":
    print("测试推荐系统...")
    recommender = FreshFoodRecommender()
    print("\n=== 测试向量搜索 ===")
    results = recommender.vector_search("脆甜苹果", limit=5)
    for r in results:
        print(f"- {r.get('product_name')}: {r.get('price')}元")
    recommender.close()
    print("\n测试完成!")