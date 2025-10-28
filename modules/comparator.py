from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer, util
import numpy as np

class LegalComparator:
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        
        print(f"Loading comparison model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("Comparison model loaded successfully")
    
    def compare_texts(self, text1: str, text2: str) -> Dict:
        
        if not text1 or not text2:
            return {
                'similarity_score': 0.0,
                'similarity_percentage': "0.0%",
                'interpretation': "One or both texts are empty",
                'differences': []
            }
        
        
        embeddings1 = self.model.encode(text1, convert_to_tensor=True)
        embeddings2 = self.model.encode(text2, convert_to_tensor=True)
        
        
        similarity = util.cos_sim(embeddings1, embeddings2).item()
        
        interpretation = self._interpret_similarity(similarity)
        
        differences = self._find_differences(text1, text2)
        
        return {
            'similarity_score': float(similarity),
            'similarity_percentage': f"{similarity * 100:.1f}%",
            'interpretation': interpretation,
            'differences': differences
        }
    
    def compare_clauses(
        self, 
        clause1: str, 
        clause2: str, 
        clause1_name: str = "Clause A",
        clause2_name: str = "Clause B"
    ) -> Dict:
        
        comparison = self.compare_texts(clause1, clause2)
        
        comparison['clause1_name'] = clause1_name
        comparison['clause2_name'] = clause2_name
        comparison['legal_implications'] = self._analyze_legal_implications(
            comparison['similarity_score']
        )
        
        return comparison
    
    def compare_documents(
        self, 
        doc1_text: str, 
        doc2_text: str,
        doc1_name: str = "Document 1",
        doc2_name: str = "Document 2"
    ) -> Dict:
        
        overall_comparison = self.compare_texts(doc1_text, doc2_text)
        
        # Section-by-section comparison
        sections1 = self._split_into_sections(doc1_text)
        sections2 = self._split_into_sections(doc2_text)
        
        section_comparisons = []
        for i, (sec1, sec2) in enumerate(zip(sections1, sections2)):
            sec_comp = self.compare_texts(sec1, sec2)
            sec_comp['section_number'] = i + 1
            section_comparisons.append(sec_comp)
        
        return {
            'document1_name': doc1_name,
            'document2_name': doc2_name,
            'overall_similarity': overall_comparison['similarity_score'],
            'overall_percentage': overall_comparison['similarity_percentage'],
            'interpretation': overall_comparison['interpretation'],
            'key_differences': overall_comparison['differences'][:10],  # Top 10
            'section_comparisons': section_comparisons,
            'legal_implications': self._analyze_legal_implications(
                overall_comparison['similarity_score']
            )
        }
    
    def find_similar_clauses(
        self, 
        query_clause: str, 
        clause_list: List[Dict[str, str]],
        top_k: int = 5
    ) -> List[Dict]:
        
        query_embedding = self.model.encode(query_clause, convert_to_tensor=True)
        
        results = []
        for clause_dict in clause_list:
            clause_text = clause_dict.get('text', '')
            clause_name = clause_dict.get('name', 'Unknown')
            
            clause_embedding = self.model.encode(clause_text, convert_to_tensor=True)
            similarity = util.cos_sim(query_embedding, clause_embedding).item()
            
            results.append({
                'name': clause_name,
                'text': clause_text,
                'similarity_score': float(similarity),
                'similarity_percentage': f"{similarity * 100:.1f}%"
            })
        
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return results[:top_k]
    
    def _interpret_similarity(self, score: float) -> str:
        """Interpret similarity score"""
        if score >= 0.95:
            return "Nearly identical - minimal or no significant differences"
        elif score >= 0.85:
            return "Very similar - minor wording differences only"
        elif score >= 0.70:
            return "Moderately similar - some semantic differences present"
        elif score >= 0.50:
            return "Somewhat similar - significant differences in content or meaning"
        else:
            return "Different - substantial differences in content and meaning"
    
    def _analyze_legal_implications(self, score: float) -> str:
        """Analyze legal implications of similarity"""
        if score >= 0.95:
            return "Clauses are essentially equivalent. No significant legal differences expected."
        elif score >= 0.85:
            return "Clauses are very similar. Minor wording differences unlikely to affect legal interpretation."
        elif score >= 0.70:
            return "Clauses share core concepts but have notable differences. Review differences carefully."
        elif score >= 0.50:
            return "Significant semantic differences exist. Different legal obligations or rights may apply."
        else:
            return "Clauses are substantially different. Likely represent different legal terms or obligations."
    
    def _find_differences(self, text1: str, text2: str) -> List[Dict]:
        
        sentences1 = [s.strip() for s in text1.split('.') if s.strip()]
        sentences2 = [s.strip() for s in text2.split('.') if s.strip()]
        
        if not sentences1 or not sentences2:
            return []
        
        differences = []
        
        embeddings1 = self.model.encode(sentences1, convert_to_tensor=True)
        embeddings2 = self.model.encode(sentences2, convert_to_tensor=True)
        
        for i, sent1 in enumerate(sentences1):
            similarities = util.cos_sim(embeddings1[i], embeddings2).cpu().numpy()
            max_sim = np.max(similarities)
            
            if max_sim < 0.7:  
                differences.append({
                    'sentence': sent1,
                    'location': 'Text 1',
                    'similarity_to_other': float(max_sim),
                    'type': 'unique_to_text1'
                })
        
        for i, sent2 in enumerate(sentences2):
            similarities = util.cos_sim(embeddings2[i], embeddings1).cpu().numpy()
            max_sim = np.max(similarities)
            
            if max_sim < 0.7:
                differences.append({
                    'sentence': sent2,
                    'location': 'Text 2',
                    'similarity_to_other': float(max_sim),
                    'type': 'unique_to_text2'
                })
        
        differences.sort(key=lambda x: x['similarity_to_other'])
        
        return differences[:15]  
    
    def _split_into_sections(self, text: str, num_sections: int = 5) -> List[str]:
        """Split text into approximately equal sections"""
        words = text.split()
        section_size = len(words) // num_sections
        
        sections = []
        for i in range(num_sections):
            start = i * section_size
            end = start + section_size if i < num_sections - 1 else len(words)
            section = ' '.join(words[start:end])
            sections.append(section)
        
        return sections
