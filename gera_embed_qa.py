import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Carrega base
with open("base_semantica.json", encoding="utf-8") as f:
    base = json.load(f)

modelo = SentenceTransformer("all-MiniLM-L6-v2")

nova_base = []
for item in base:
    texto = item.get("pergunta") or item.get("conteudo")
    embedding = modelo.encode(texto).tolist()
    nova_base.append({
        "pergunta": item.get("pergunta"),
        "resposta": item.get("resposta") or item.get("conteudo"),
        "embedding": embedding
    })

with open("base_embed_qa.json", "w", encoding="utf-8") as f:
    json.dump(nova_base, f, ensure_ascii=False, indent=2)

print("✅ Embeddings locais gerados com sucesso.")
