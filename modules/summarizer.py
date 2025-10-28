"""Legal document summarization module"""
from typing import Dict, List, Optional
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class LegalSummarizer:
    """Summarize legal documents using BART/T5"""
    
    def __init__(
        self, 
        model_name: str = "facebook/bart-large-cnn",
        max_length: int = 1024,
        min_length: int = 100
    ):
        
        self.model_name = model_name
        self.max_length = max_length
        self.min_length = min_length
        self.summarizer = None
        
        print(f"Loading summarization model: {model_name}...")
        self._load_model()
    
    def _load_model(self):
        try:
            self.summarizer = pipeline(
                "summarization",
                model=self.model_name,
                torch_dtype=torch.float32,
                device=-1  
            )
            print("Summarization model loaded successfully")
        except Exception as e:
            print(f"Error loading summarization model: {e}")
            raise
    
    def summarize_text(
        self, 
        text: str, 
        max_length: Optional[int] = None,
        min_length: Optional[int] = None
    ) -> str:
       
        if not text or not text.strip():
            return "No text provided for summarization"
        
        max_len = max_length or self.max_length
        min_len = min_length or self.min_length
        
        try:
            if len(text) > 10000:
                text = text[:10000]  
            
            summary = self.summarizer(
                text,
                max_length=max_len,
                min_length=min_len,
                do_sample=False,
                truncation=True
            )
            
            return summary[0]['summary_text']
            
        except Exception as e:
            return f"Error during summarization: {str(e)}"
    
    def summarize_legal_judgment(self, text: str) -> Dict[str, str]:
        
        sections = self._extract_sections(text)
        
        summary_dict = {
            'key_facts': self._summarize_section(
                sections.get('facts', text[:2000]),
                "Summarize the key facts of this legal case:",
                max_length=150
            ),
            'decision': self._summarize_section(
                sections.get('decision', text[-2000:]),
                "Summarize the court's decision:",
                max_length=150
            ),
            'legal_basis': self._summarize_section(
                sections.get('legal_basis', text),
                "What is the legal basis for this judgment?",
                max_length=150
            ),
            'precedent': self._extract_precedents(text),
            'verdict': self._extract_verdict(text)
        }
        
        return summary_dict
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract different sections from legal document"""
        sections = {}
        
        text_lower = text.lower()
        
        if 'facts' in text_lower:
            start = text_lower.index('facts')
            sections['facts'] = text[start:start+2000]
        
        if 'decision' in text_lower or 'judgment' in text_lower:
            if 'decision' in text_lower:
                start = text_lower.index('decision')
            else:
                start = text_lower.index('judgment')
            sections['decision'] = text[start:start+2000]
        
        return sections
    
    def _summarize_section(
        self, 
        text: str, 
        prefix: str, 
        max_length: int = 150
    ) -> str:
        section_text = f"{prefix}\n\n{text[:1500]}"
        return self.summarize_text(section_text, max_length=max_length, min_length=30)
    
    def _extract_precedents(self, text: str) -> str:
        keywords = ['precedent', 'cited', 'referred', 'vs.', 'v.']
        
        sentences = []
        for line in text.split('\n'):
            line_lower = line.lower()
            if any(kw in line_lower for kw in keywords):
                sentences.append(line.strip())
                if len(sentences) >= 3:
                    break
        
        if sentences:
            return '; '.join(sentences[:3])
        else:
            return "No specific precedents identified in the text"
    
    def _extract_verdict(self, text: str) -> str:
        
        keywords = ['held', 'ordered', 'directed', 'dismissed', 'allowed', 'verdict']
        
        text_lower = text.lower()
        for keyword in keywords:
            if keyword in text_lower:
                start = max(0, text_lower.index(keyword) - 100)
                end = min(len(text), text_lower.index(keyword) + 200)
                return text[start:end].strip()
        
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        if paragraphs:
            return paragraphs[-1][:300]
        
        return "Verdict not clearly identified"
    
    def batch_summarize(self, texts: List[str]) -> List[str]:
        
        summaries = []
        for text in texts:
            summary = self.summarize_text(text)
            summaries.append(summary)
        
        return summaries
