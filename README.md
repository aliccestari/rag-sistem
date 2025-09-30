# Sistema RAG Básico

Um sistema básico de Retrieval-Augmented Generation (RAG) implementado seguindo as melhores práticas da documentação oficial da Microsoft, Hugging Face e MongoDB.

## Instalação

1. Clone ou baixe os arquivos
2. Instale as dependências:

```bash
pip3 install -r requirements.txt
```

## Arquivos

- `rag_oficial.py` - Sistema RAG principal (implementação oficial)
- `rag_simples.py` - Versão anterior (para comparação)
- `requirements.txt` - Dependências do projeto
- `README.md` - Este arquivo

## Como usar

### Exemplo básico

```python
from rag_oficial import RAGBasico

# Cria o sistema RAG
rag = RAGBasico()

# Seus documentos
documentos = [
    "Python é uma linguagem de programação popular.",
    "Machine Learning é um campo da IA.",
    "TensorFlow é uma biblioteca para ML."
]

# Adiciona documentos
rag.adicionar_documentos(documentos)

# Faz uma pergunta
resposta = rag.perguntar("O que é Python?")
print(resposta)
```

### Executar exemplos

```bash
# Sistema RAG oficial (recomendado)
python3 rag_oficial.py

# Sistema anterior (para comparação)
python3 rag_simples.py
```

## Componentes

1. **SentenceTransformer**: Gera embeddings dos documentos e consultas
2. **FAISS Index**: Índice vetorial para busca eficiente por similaridade
3. **Retriever**: Encontra documentos mais relevantes para a consulta
4. **Response Generator**: Retorna o documento mais relevante como resposta

## Como funciona

1. **Indexação**: Documentos são convertidos em embeddings e armazenados no índice FAISS
2. **Recuperação**: A pergunta é convertida em embedding e documentos similares são encontrados
3. **Resposta**: O documento mais relevante é retornado como resposta

## Configuração

Você pode personalizar o modelo de embeddings:

```python
# Modelo mais preciso (maior, mais lento)
rag = RAGBasico(embedding_model="all-mpnet-base-v2")

# Modelo mais rápido (menor, mais rápido)
rag = RAGBasico(embedding_model="all-MiniLM-L6-v2")
```

## Personalização

- Substitua os documentos pelos seus próprios
- Ajuste o número de documentos recuperados (parâmetro `k`)
- Use diferentes modelos de embeddings
- Adicione mais documentos à base de conhecimento

## Exemplo de saída

```
Pergunta: O que é Python?

Documentos encontrados:
  1. Score: 0.856 - Python é uma linguagem de programação de alto nível, interpretada e de propósito geral...
  2. Score: 0.743 - Machine Learning é um subcampo da inteligência artificial...
  3. Score: 0.621 - TensorFlow é uma biblioteca de código aberto...

Resposta: Python é uma linguagem de programação de alto nível, interpretada e de propósito geral.
```

## Requisitos

- Python 3.8+
- ~1GB de RAM
- Conexão com internet (para baixar modelos na primeira execução)

## Troubleshooting

- **Erro de memória**: Reduza o número de documentos ou use modelos menores
- **Modelos não carregam**: Verifique sua conexão com internet
- **Respostas ruins**: Adicione mais documentos relevantes ou ajuste os parâmetros

## Próximos passos

- Adicione mais documentos à base de conhecimento
- Experimente diferentes modelos de embeddings
- Implemente persistência de dados
- Adicione interface web
- Melhore a qualidade das respostas

## Referências

- [Microsoft RAG Solution Design Guide](https://learn.microsoft.com/pt-br/azure/architecture/ai-ml/guide/rag/rag-solution-design-and-evaluation-guide)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [MongoDB Vector Search](https://mongodb.com/pt-br/docs/atlas/atlas-vector-search/tutorials/local-rag)
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS Documentation](https://faiss.ai/)
