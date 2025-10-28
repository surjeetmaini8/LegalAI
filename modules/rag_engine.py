"""RAG-based Q&A engine for legal documents"""
from typing import List, Dict, Optional
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

class RAGEngine:
    
    
    def __init__(
        self, 
        model_name: str = "google/flan-t5-large",
        max_new_tokens: int = 512,
        temperature: float = 0.3
    ):
        
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.llm = None
        self.qa_chain = None
        
        print(f"Loading LLM: {model_name}...")
        self._load_llm()
    
    def _load_llm(self):
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  
                low_cpu_mem_usage=True
            )
            
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.95,
                top_k=50
            )
            
            self.llm = HuggingFacePipeline(pipeline=pipe)
            print("LLM loaded successfully")
            
        except Exception as e:
            print(f"Error loading LLM: {e}")
            raise
    
    def create_qa_chain(self, retriever, prompt_template: Optional[str] = None):
        
        if prompt_template is None:
            template = """You are a legal AI assistant. Use the following legal document context to answer the question. 
If you don't know the answer based on the context, say "I cannot find sufficient information in the provided legal documents to answer this question."

Context from legal documents:
{context}

Question: {question}

Provide a clear, accurate answer based on the legal context above. Include relevant legal provisions, sections, or case references if mentioned in the context."""

            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
        else:
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
        
        # Create RetrievalQA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        print("QA chain created successfully")
    
    def query(self, question: str) -> Dict:
        
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized. Call create_qa_chain() first.")
        
        try:
            result = self.qa_chain({"query": question})
            
            answer = result.get('result', 'No answer generated')
            source_docs = result.get('source_documents', [])
            
            sources = []
            for doc in source_docs:
                sources.append({
                    'source': doc.metadata.get('source', 'Unknown'),
                    'chunk_id': doc.metadata.get('chunk_id', 'N/A'),
                    'content': doc.page_content[:200] + "..."  # Preview
                })
            
            return {
                'answer': answer,
                'sources': sources,
                'question': question
            }
            
        except Exception as e:
            print(f"Error during query: {e}")
            return {
                'answer': f"Error processing question: {str(e)}",
                'sources': [],
                'question': question
            }
    
    def batch_query(self, questions: List[str]) -> List[Dict]:
        
        results = []
        for question in questions:
            result = self.query(question)
            results.append(result)
        
        return results
    
    def get_direct_answer(self, context: str, question: str) -> str:
        
        if self.llm is None:
            raise ValueError("LLM not initialized")
        
        prompt = f"""Based on the following legal context, answer the question:

Context: {context}

Question: {question}

Answer:"""
        
        try:
            answer = self.llm(prompt)
            return answer
        except Exception as e:
            return f"Error generating answer: {str(e)}"
