"""Vector store management using FAISS"""
import os
from pathlib import Path
from typing import List, Optional
import pickle
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

class VectorStoreManager:
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        
        print(f"Loading embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},  # Use 'cuda' if GPU available
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store = None
    
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        
        if not documents:
            raise ValueError("No documents provided to create vector store")
        
        print(f"Creating vector store from {len(documents)} documents...")
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        print("Vector store created successfully")
        
        return self.vector_store
    
    def save_vector_store(self, save_path: str):
        
        if self.vector_store is None:
            raise ValueError("No vector store to save")
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.vector_store.save_local(str(save_path))
        print(f"Vector store saved to {save_path}")
    
    def load_vector_store(self, load_path: str) -> FAISS:
        
        load_path = Path(load_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Vector store not found at {load_path}")
        
        print(f"Loading vector store from {load_path}...")
        self.vector_store = FAISS.load_local(
            str(load_path),
            self.embeddings,
            allow_dangerous_deserialization=True 
        )
        print("Vector store loaded successfully")
        
        return self.vector_store
    
    def add_documents(self, documents: List[Document]):
        
        if self.vector_store is None:
            raise ValueError("No vector store initialized. Create one first.")
        
        print(f"Adding {len(documents)} documents to vector store...")
        self.vector_store.add_documents(documents)
        print("Documents added successfully")
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5,
        filter_dict: Optional[dict] = None
    ) -> List[Document]:
        
        if self.vector_store is None:
            raise ValueError("No vector store initialized")
        
        results = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter_dict
        )
        
        return results
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 5
    ) -> List[tuple]:
        
        if self.vector_store is None:
            raise ValueError("No vector store initialized")
        
        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=k
        )
        
        return results
    
    def get_retriever(self, search_kwargs: Optional[dict] = None):
        
        if self.vector_store is None:
            raise ValueError("No vector store initialized")
        
        if search_kwargs is None:
            search_kwargs = {'k': 5}
        
        return self.vector_store.as_retriever(
            search_kwargs=search_kwargs
        )
