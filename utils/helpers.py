"""Utility helper functions"""
import os
from pathlib import Path
from typing import List, Dict
import streamlit as st

def get_document_list(documents_dir: str) -> List[str]:
    """Get list of all documents in the documents directory"""
    supported_formats = ['.pdf', '.docx', '.txt']
    documents = []
    
    doc_path = Path(documents_dir)
    if doc_path.exists():
        for file in doc_path.iterdir():
            if file.suffix.lower() in supported_formats:
                documents.append(file.name)
    
    return sorted(documents)

def format_sources(sources: List[Dict]) -> str:
    if not sources:
        return "No sources found."
    
    formatted = "\n\n Sources:**\n"
    for i, source in enumerate(sources, 1):
        doc_name = source.get('source', 'Unknown')
        page = source.get('page', 'N/A')
        formatted += f"{i}. {doc_name} (Page: {page})\n"
    
    return formatted

def truncate_text(text: str, max_length: int = 500) -> str:
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def display_message(role: str, content: str, avatar: str = None):
    avatars = {
        'user': '👤',
        'assistant': '🤖',
        'system': 'ℹ️'
    }
    
    with st.chat_message(role, avatar=avatar or avatars.get(role, '💬')):
        st.markdown(content)

def create_summary_points(summary_dict: Dict) -> str:
    points = []
    
    if 'key_facts' in summary_dict:
        points.append(f"**Key Facts:** {summary_dict['key_facts']}")
    
    if 'decision' in summary_dict:
        points.append(f"**Decision:** {summary_dict['decision']}")
    
    if 'legal_basis' in summary_dict:
        points.append(f"**Legal Basis:** {summary_dict['legal_basis']}")
    
    if 'precedent' in summary_dict:
        points.append(f"**Precedent Used:** {summary_dict['precedent']}")
    
    if 'verdict' in summary_dict:
        points.append(f"**Verdict:** {summary_dict['verdict']}")
    
    return "\n\n".join(points) if points else summary_dict.get('text', '')

def calculate_similarity_percentage(score: float) -> str:
    percentage = score * 100
    return f"{percentage:.1f}%"

def get_similarity_emoji(score: float) -> str:
    """Get emoji based on similarity score"""
    if score >= 0.9:
        return "🟢"  
    elif score >= 0.7:
        return "🟡" 
    elif score >= 0.5:
        return "🟠"  
    else:
        return "🔴" 
