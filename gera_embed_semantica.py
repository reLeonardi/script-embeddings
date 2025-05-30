import json
from sentence_transformers import SentenceTransformer

"""
Script para gerar embeddings locais da base semântica (conteúdo livre).

Entrada: base_semantica.json (lista de itens com 'conteudo' e 'fonte')
Saída: base_embed_semantica.json (mesmo conteúdo com embedding adicional)
"""

# Carrega base semântica
with open("base_semantica.json", encoding="utf-8") as f:
    base = json.load(f)

# Modelo de embedding
modelo = SentenceTransformer("all-MiniLM-L6-v2")

nova_base = []
for item in base:
    texto = item.get("conteudo")
    embedding = modelo.encode(texto).tolist()
    nova_base.append({
        "conteudo": texto,
        "fonte": item.get("fonte"),
        "embedding": embedding
    })

# Salva embeddings
with open("base_embed_semantica.json", "w", encoding="utf-8") as f:
    json.dump(nova_base, f, ensure_ascii=False, indent=2)

print("✅ Embeddings da base semântica gerados com sucesso.")
