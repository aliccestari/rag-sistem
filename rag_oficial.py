#!/usr/bin/env python3
"""
RAG Básico - Implementação baseada na documentação oficial
Somente recupera os documentos mais relevantes de uma base usando embeddings e FAISS e retorna o documento mais próximo como resposta.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Tuple

class RAGBasico:
    """
    Sistema RAG básico e funcional
    Implementa as melhores práticas da documentação oficial
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Inicializa o sistema RAG
        
        Args:
            embedding_model: Modelo de embeddings a ser usado
        """
        print("Inicializando sistema RAG...")
        self.embedding_model = SentenceTransformer(embedding_model) # cria o modelo de embeddings, converte texto em vetores
        self.documents: List[str] = [] # lista de documentos
        self.embeddings: np.ndarray = None # vetores de embeddings
        self.index: faiss.Index = None # índice de busca
        print("Sistema RAG inicializado!")
    
    def adicionar_documentos(self, documentos: List[str]) -> None:
        """
        Adiciona documentos ao sistema e cria o índice de busca
        
        Args:
            documentos: Lista de documentos para indexar
        """
        print(f"Processando {len(documentos)} documentos...") # imprime o número de documentos
        
        # Armazena os documentos
        self.documents = documentos
        
        # Gera embeddings para todos os documentos
        print("Gerando embeddings...")
        self.embeddings = self.embedding_model.encode(documentos, convert_to_tensor=False)
        
        # Cria índice FAISS para busca eficiente
        print("Criando índice de busca...")
        dimension = self.embeddings.shape[1] # obtem a dimensao dos embeddings, define o tamanho do indice
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidiana), busca por similaridade
        self.index.add(self.embeddings.astype('float32')) # adiciona os embeddings ao indice, torna a busca possivel, float32 é o formato esperado pelo FAISS
        
        print(f"Índice criado com {len(documentos)} documentos!")
    
    def buscar(self, consulta: str, k: int = 3) -> Tuple[List[str], List[float]]:
        """
        Busca documentos mais relevantes para uma consulta
        
        Args:
            consulta: Pergunta ou consulta do usuário
            k: Número de documentos para retornar
            
        Returns:
            Tupla com (documentos_encontrados, scores)
        """
        if self.index is None:
            raise ValueError("Nenhum documento foi adicionado ao sistema. Use adicionar_documentos() primeiro.")
        
        print(f"Buscando documentos para: '{consulta}'")
        
        # Gera embedding da consulta, converte a pergunta em vetor
        consulta_embedding = self.embedding_model.encode([consulta], convert_to_tensor=False)
        
        # Busca documentos mais similares, scores = distancias e indices = posicoes
        scores, indices = self.index.search(consulta_embedding.astype('float32'), k)
        
        # Retorna documentos e scores
        documentos_encontrados = [self.documents[idx] for idx in indices[0]]
        scores_lista = scores[0].tolist()
        
        return documentos_encontrados, scores_lista
    
    def perguntar(self, pergunta: str, k: int = 3) -> str:
        """
        Faz uma pergunta e retorna a resposta baseada nos documentos
        
        Args:
            pergunta: Pergunta do usuário
            k: Número de documentos para considerar
            
        Returns:
            Resposta baseada nos documentos mais relevantes
        """
        print(f"\nPergunta: {pergunta}")
        
        # Busca documentos relevantes
        documentos, scores = self.buscar(pergunta, k)
        
        # Mostra resultados
        print(f"\nDocumentos encontrados:")
        for i, (doc, score) in enumerate(zip(documentos, scores)):
            print(f"  {i+1}. Score: {score:.3f} - {doc[:80]}...")
        
        # Retorna o documento mais relevante
        if documentos:
            resposta = documentos[0]
            print(f"\nResposta: {resposta}")
            return resposta
        else:
            print("\nNenhum documento relevante encontrado.")
            return "Nenhum documento relevante encontrado."

def main():
    """Exemplo de uso do sistema RAG"""
    print("=" * 60)
    print("SISTEMA RAG BÁSICO - IMPLEMENTAÇÃO OFICIAL")
    print("=" * 60)
    
    # Cria instância do RAG
    rag = RAGBasico()
    
    # Documentos de exemplo (base de conhecimento)
    documentos = [
        "Python é uma linguagem de programação de alto nível, interpretada e de propósito geral.",
        "Machine Learning é um subcampo da inteligência artificial que se concentra em algoritmos que podem aprender com dados.",
        "TensorFlow é uma biblioteca de código aberto para machine learning desenvolvida pelo Google.",
        "PyTorch é uma biblioteca de machine learning de código aberto baseada no Torch.",
        "Deep Learning é uma subárea do machine learning que usa redes neurais com múltiplas camadas.",
        "Inteligência artificial é a simulação de inteligência humana em máquinas.",
        "Processamento de linguagem natural (NLP) é um campo da IA que lida com a interação entre computadores e linguagem humana.",
        "Redes neurais são sistemas computacionais inspirados em redes neurais biológicas."
    ]
    
    # Adiciona documentos ao sistema
    rag.adicionar_documentos(documentos)
    
    # Perguntas de teste
    perguntas = [
        "O que é Python?",
        "Como funciona o machine learning?",
        "Qual a diferença entre TensorFlow e PyTorch?",
        "O que são redes neurais?",
        "O que é deep learning?"
    ]
    
    print("\n" + "=" * 60)
    print("TESTANDO O SISTEMA")
    print("=" * 60)
    
    # Testa cada pergunta
    for pergunta in perguntas:
        rag.perguntar(pergunta)
        print("-" * 60)

if __name__ == "__main__":
    main()
