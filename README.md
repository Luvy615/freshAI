# 生小鲜 - 智能生鲜电商推荐系统

基于LangChain Agent的多模态生鲜电商对话式推荐系统，支持：

- 文本对话搜索商品
- 上传图片搜索商品（CLIP多模态向量检索）
- 知识图谱（Neo4j）+ 向量数据库（Milvus）双通道检索
- DeepSeek/Qwen VL大模型对话生成

## 项目结构

```
生小鲜/
├── demo/
│   ├── fresh_food_recommender.py   # 推荐系统核心（Neo4j + Milvus客户端）
│   └── fresh_food_agent.py     # Agent智能代理
├── app/
│   └── main.py              # Streamlit Web界面
├── .streamlit/
│   ├── config.toml        # Streamlit配置
│   └── secrets.toml     # API密钥配置（部署用）
├── requirements.txt
└── README.md
```

## 本地运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行Streamlit
streamlit run app/main.py
```

## 部署到Streamlit Cloud

1. 将代码推送到GitHub仓库
2. 在Streamlit Cloud创建新应用，选择该仓库
3. 在Settings中配置Secrets：

```toml
# secrets.toml
NEO4J_URI = "neo4j+s://9acccd6e.databases.neo4j.io"
NEO4J_USER = "9acccd6e"
NEO4J_PASSWORD = "FBPVbuhBV8LewZXyGbIfKIMucDx4V0gjDrwMaNYfvh4"

MILVUS_URI = "https://in03-c690e46590549f3.serverless.gcp-us-west1.cloud.zilliz.com"
MILVUS_TOKEN = "b78e04209431b99f6ea295cd538761f97b4b2101fbba8ce281816a1454a176e4404a4a332b1a82f33a741f3ed0cba38f9379b0f8"

DEEPSEEK_API_KEY = "sk-xxx"
QWEN_VL_API_KEY = "sk-xxx"
```

4. 部署完成！

## 环境要求

- Python 3.10+
- Neo4j Aura（已配置）
- Milvus/Zilliz Cloud（已配置）
- DeepSeek API
- Qwen VL Plus API

## 使用说明

1. 在对话框中输入想要购买的生鲜商品（如"我想买苹果"）
2. 也可以上传商品图片进行搜索
3. 系统会在知识图谱和向量数据库中双通道搜索
4. 以对话形式展示推荐商品