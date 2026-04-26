# -*- coding: utf-8 -*-
"""
生鲜电商智能Agent
- 意图理解
- 多模态融合
- 对话管理
"""

import os
import sys
import io
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# 添加项目根目录到路径
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

from demo.fresh_food_recommender import (
    FreshFoodRecommender, llm_generate, llm_intent_recognition,
    extract_entities_from_image, clean_str
)


# ==================== 规则匹配 ====================
PRODUCT_KEYWORDS = [
    '苹果', '香蕉', '荔枝', '樱桃', '草莓', '橙子', '梨', '桃子', '西瓜', '哈密瓜',
    '芒果', '榴莲', '火龙果', '猕猴桃', '菠萝', '葡萄', '柚子', '柠檬', '石榴', '枇杷',
    '白菜', '菠菜', '西红柿', '黄瓜', '土豆', '胡萝卜', '洋葱', '大蒜', '生姜', '辣椒',
    '茄子', '豆角', '芹菜', '生菜', '油菜', '韭菜', '茼蒿', '香菜', '萝卜', '莲藕',
    '猪肉', '牛肉', '羊肉', '鸡肉', '鸭肉', '鹅肉', '排骨', '五花肉', '里脊肉', '牛排',
    '鱼', '虾', '螃蟹', '贝类', '海参', '鲍鱼', '鱿鱼', '章鱼', '带鱼', '黄鱼',
    '豆腐', '豆皮', '豆芽', '豆浆', '腐竹', '、素鸡', '千张',
    '鸡蛋', '鸭蛋', '鹅蛋', '鹌鹑蛋', '皮蛋', '咸蛋',
    '牛奶', '酸奶', '奶酪', '黄油', '奶油', '奶粉'
]

TASTE_KEYWORDS = ['脆', '甜', '酸', '软', '鲜', '嫩', '多汁', '香', '辣', '麻', '咸', '清淡', '浓郁']

CATEGORY_KEYWORDS = {
    '水果': ['苹果', '香蕉', '荔枝', '樱桃', '草莓', '橙子', '梨', '桃子', '西瓜', '哈密瓜', '芒果', '榴莲'],
    '蔬菜': ['白菜', '菠菜', '西红柿', '黄瓜', '土豆', '胡萝卜', '洋葱', '大蒜', '生姜', '辣椒'],
    '肉禽': ['猪肉', '牛肉', '羊肉', '鸡肉', '鸭肉', '鹅肉', '排骨', '五花肉'],
    '海鲜': ['鱼', '虾', '螃蟹', '贝类', '海参', '鲍鱼', '鱿鱼', '章鱼'],
    '豆制品': ['豆腐', '豆皮', '豆芽', '豆浆', '腐竹'],
    '禽蛋': ['鸡蛋', '鸭蛋', '鹅蛋', '鹌鹑蛋'],
    '奶制品': ['牛奶', '酸奶', '奶酪', '黄油']
}

SCENE_KEYWORDS = {
    '日常食用': ['日常', '平时', '家常'],
    '送礼': ['送礼', '送人', '礼盒'],
    '宴请': ['宴请', '招待', '客人'],
    '宝宝辅食': ['宝宝', '婴儿', '小孩'],
    '减脂': ['减肥', '减脂', '健身'],
    '节日': ['节日', '过年', '春节', '中秋']
}


class FreshFoodAgent:
    def __init__(self):
        self.recommender = FreshFoodRecommender()
        self.conversation_history = []
        self.current_image = None
        self.user_preferences = {}
    
    def _rule_intent(self, user_input: str) -> Dict[str, Any]:
        """规则匹配意图"""
        product_type = ""
        tastes = []
        category = ""
        scene = "日常食用"
        
        for kw in PRODUCT_KEYWORDS:
            if kw in user_input:
                product_type = kw
                break
        
        for kw in TASTE_KEYWORDS:
            if kw in user_input:
                tastes.append(kw)
        
        for cat, keywords in CATEGORY_KEYWORDS.items():
            for kw in keywords:
                if kw in user_input:
                    category = cat
                    if not product_type:
                        product_type = kw
                    break
        
        for sc, keywords in SCENE_KEYWORDS.items():
            for kw in keywords:
                if kw in user_input:
                    scene = sc
                    break
        
        return {
            "product_type": product_type,
            "tastes": tastes,
            "category": category,
            "scene": scene
        }
    
    def think(self, user_input: str, image_path: str = None) -> Dict[str, Any]:
        """Agent核心思考流程"""
        # 1. 意图理解
        if image_path and os.path.exists(image_path):
            self.current_image = image_path
            img_entities = extract_entities_from_image(image_path)
            llm_entities = llm_intent_recognition(user_input)
            intent = {**img_entities, **llm_entities}
            intent["has_image"] = True
        else:
            intent = self._rule_intent(user_input)
            llm_entities = llm_intent_recognition(user_input)
            for k, v in llm_entities.items():
                if not intent.get(k) or (isinstance(v, list) and not v):
                    intent[k] = v
            intent["has_image"] = False
        
        # 2. 搜索执行
        product_type = intent.get("product_type", "")
        taste = intent.get("tastes", [""])[0] if intent.get("tastes") else ""
        category = intent.get("category", "")
        
        kg_results = []
        vec_results = []
        
        # 优先使用知识图谱搜索（更精确）
        if product_type or category:
            kg_results = self.recommender.kg_search(
                product_type=product_type if product_type else None,
                category=category if category else None,
                taste=taste if taste else None,
                limit=15
            )
        
        # 如果知识图谱结果不足，再用向量搜索补充
        if len(kg_results) < 5:
            query_text = user_input
            if product_type:
                query_text = f"{product_type} {taste}" if taste else product_type
            vec_results = self.recommender.vector_search(query_text, limit=10)
        
        # 图像搜索
        if self.current_image and os.path.exists(self.current_image):
            img_results = self.recommender._search_by_image(self.current_image, limit=10)
            if img_results:
                vec_results = img_results
        
        # 3. 结果合并
        merged_results = self.recommender.merge_results(kg_results, vec_results, limit=10)
        
        # 4. 保存偏好
        if product_type:
            self.user_preferences["last_product_type"] = product_type
        if taste:
            self.user_preferences["last_taste"] = taste
        if category:
            self.user_preferences["last_category"] = category
        
        return {
            "intent": intent,
            "results": merged_results,
            "kg_results": kg_results,
            "vec_results": vec_results
        }
    
    def run(self, user_input: str, image_path: str = None) -> str:
        """执行完整流程并生成回复"""
        # 执行搜索
        result_data = self.think(user_input, image_path)
        intent = result_data["intent"]
        products = result_data["results"]
        
        # 构建商品列表文本
        product_lines = []
        for i, p in enumerate(products[:6], 1):
            name = p.get("productName", p.get("product_name", ""))
            price = p.get("price", 0)
            shop = p.get("shop", "")
            taste = p.get("taste", "")
            line = f"{i}. {name}\n   💰 {price}元"
            if taste:
                line += f" | {taste}"
            if shop:
                line += f" | 🏪 {shop}"
            product_lines.append(line)
        
        products_text = "\n\n".join(product_lines) if product_lines else "暂无推荐商品"
        
        # 生成回复
        prompt = f"""用户说：{user_input}

我找到了 {len(products)} 款合适的生鲜商品推荐给你：

{products_text}

请用友好的语气回复用户，简短介绍推��理由，并询问是否需要查看更多或有什么其他要求。"""
        
        history = self.conversation_history[-10:] if self.conversation_history else []
        response = llm_generate(prompt, history)
        
        # 保存对话历史
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        return response
    
    def get_products(self, user_input: str, image_path: str = None) -> List[Dict]:
        """获取推荐商品列表"""
        result_data = self.think(user_input, image_path)
        return result_data["results"]
    
    def clear_image(self):
        """清除当前图片"""
        self.current_image = None
    
    def clear_history(self):
        """清除对话历史"""
        self.conversation_history = []
    
    def close(self):
        """资源清理"""
        self.recommender.close()


def format_products_display(products: List[Dict]) -> str:
    """格式化商品列表显示"""
    if not products:
        return "暂无推荐商品"
    
    lines = []
    for i, p in enumerate(products, 1):
        name = clean_str(p.get("productName", p.get("product_name", "")))
        price = p.get("price", 0)
        sales = p.get("sales", 0)
        taste = clean_str(p.get("taste", ""))
        source = p.get("source", "")
        
        line = f"**{i}. {name}**\n"
        line += f"   - 💰 价格: {price}元\n"
        line += f"   - 📦 销量: {sales}件\n"
        if taste:
            line += f"   - 🍯 口感: {taste}\n"
        line += f"   - 🔍 来源: {source}"
        
        lines.append(line)
    
    return "\n\n".join(lines)


if __name__ == "__main__":
    print("生鲜电商智能Agent测试...")
    print("=" * 50)
    
    agent = FreshFoodAgent()
    
    # 测试对话
    test_inputs = [
        "我想买一些苹果",
        "有没有脆甜的苹果推荐",
        "给宝宝买水果"
    ]
    
    for input_text in test_inputs:
        print(f"\n用户: {input_text}")
        response = agent.run(input_text)
        print(f"\n助手: {response}")
        print("-" * 50)
    
    agent.close()
    print("\n测试完成!")