

import streamlit as st
import os
from pathlib import Path
import tempfile
from datetime import datetime

# Import custom modules
from config import Config
from modules.document_processor import DocumentProcessor
from modules.vector_store import VectorStoreManager
from modules.rag_engine import RAGEngine
from modules.summarizer import LegalSummarizer
from modules.comparator import LegalComparator
from modules.voice_handler import VoiceHandler
from utils.helpers import (
    get_document_list, 
    format_sources, 
    display_message,
    create_summary_points,
    calculate_similarity_percentage,
    get_similarity_emoji
)

st.set_page_config(
    page_title="Legal AI Chat Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #64748B;
        margin-bottom: 2rem;
    }
    .feature-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #F1F5F9;
        margin: 1rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.vector_store_loaded = False
        st.session_state.chat_history = []
        st.session_state.current_mode = "Q&A"
        
        st.session_state.doc_processor = None
        st.session_state.vector_manager = None
        st.session_state.rag_engine = None
        st.session_state.summarizer = None
        st.session_state.comparator = None
        st.session_state.voice_handler = None

def load_components():
    with st.spinner("Loading AI models... This may take a few minutes on first run."):
        try:
            Config.ensure_directories()
            
            if st.session_state.doc_processor is None:
                st.session_state.doc_processor = DocumentProcessor(
                    chunk_size=Config.MAX_CHUNK_SIZE,
                    chunk_overlap=Config.CHUNK_OVERLAP
                )
            
            vector_store_path = Config.get_vector_store_path()
            vector_store_files = list(Path(vector_store_path).glob("*"))
            
            if st.session_state.vector_manager is None:
                st.session_state.vector_manager = VectorStoreManager(
                    embedding_model=Config.EMBEDDING_MODEL
                )
            
         
            if vector_store_files and not st.session_state.vector_store_loaded:
                try:
                    st.session_state.vector_manager.load_vector_store(vector_store_path)
                    st.session_state.vector_store_loaded = True
                    st.success("✅ Loaded existing vector store")
                except Exception as e:
                    st.warning(f"Could not load vector store: {e}. Will create new one when documents are added.")
            
            st.session_state.initialized = True
            
        except Exception as e:
            st.error(f"Error loading components: {e}")
            return False
    
    return True

def load_rag_engine():
   
    if st.session_state.rag_engine is None:
        with st.spinner("Loading Q&A engine..."):
            st.session_state.rag_engine = RAGEngine(
                model_name=Config.LLM_MODEL,
                max_new_tokens=Config.MAX_NEW_TOKENS,
                temperature=Config.TEMPERATURE
            )
            
            if st.session_state.vector_store_loaded:
                retriever = st.session_state.vector_manager.get_retriever(
                    search_kwargs={'k': Config.TOP_K_RESULTS}
                )
                st.session_state.rag_engine.create_qa_chain(retriever)

def load_summarizer():
    
    if st.session_state.summarizer is None:
        with st.spinner("Loading summarization model..."):
            st.session_state.summarizer = LegalSummarizer(
                model_name=Config.SUMMARIZATION_MODEL
            )

def load_comparator():
   
    if st.session_state.comparator is None:
        with st.spinner("Loading comparison model..."):
            st.session_state.comparator = LegalComparator(
                model_name=Config.EMBEDDING_MODEL
            )

def load_voice_handler():
    
    if st.session_state.voice_handler is None:
        with st.spinner("Loading voice models..."):
            st.session_state.voice_handler = VoiceHandler(
                whisper_model=Config.WHISPER_MODEL,
                tts_lang=Config.TTS_LANG
            )

def sidebar():
    
    with st.sidebar:
        st.markdown("Legal AI Assistant")
        st.markdown("---")
        
        st.session_state.current_mode = st.radio(
            "Select Mode",
            ["Q&A", "Summarization", "Clause Comparison", "Voice Mode"],
            index=0
        )
        
        st.markdown("---")
        
       
        st.markdown(" Document Management")
        
        
        uploaded_files = st.file_uploader(
            "Upload Legal Documents",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Upload PDF, DOCX, or TXT files"
        )
        
        if uploaded_files:
            if st.button("Process Uploaded Documents"):
                process_uploaded_documents(uploaded_files)
        
   
        docs = get_document_list(Config.get_documents_path())
        if docs:
            st.markdown(f"**Indexed Documents:** {len(docs)}")
            with st.expander("View Documents"):
                for doc in docs:
                    st.text(f" {doc}")
        else:
            st.info("No documents indexed yet. Upload documents to get started!")
        
        st.markdown("---")
        
    
        with st.expander("ℹ️ System Info"):
            st.markdown(f"""
            **Models:**
            - LLM: {Config.LLM_MODEL.split('/')[-1]}
            - Embeddings: {Config.EMBEDDING_MODEL.split('/')[-1]}
            - Summarization: {Config.SUMMARIZATION_MODEL.split('/')[-1]}
            
            **Status:**
            - Vector Store: {'Loaded' if st.session_state.vector_store_loaded else '❌ Not Loaded'}
            - Documents: {len(docs)} files
            """)

def process_uploaded_documents(uploaded_files):
    """Process and index uploaded documents"""
    with st.spinner(f"Processing {len(uploaded_files)} document(s)..."):
        try:
          
            docs_path = Config.get_documents_path()
            saved_files = []
            
            for uploaded_file in uploaded_files:
                file_path = Path(docs_path) / uploaded_file.name
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                saved_files.append(str(file_path))
                st.success(f"Saved: {uploaded_file.name}")
           
            chunked_docs = st.session_state.doc_processor.process_directory(docs_path)
            
            if chunked_docs:
               
                if st.session_state.vector_store_loaded:
                    st.session_state.vector_manager.add_documents(chunked_docs)
                else:
                    st.session_state.vector_manager.create_vector_store(chunked_docs)
                    st.session_state.vector_store_loaded = True
                
                st.session_state.vector_manager.save_vector_store(
                    Config.get_vector_store_path()
                )
                
                st.success(f"Indexed {len(chunked_docs)} chunks from {len(uploaded_files)} document(s)")
                
                if st.session_state.rag_engine is not None:
                    retriever = st.session_state.vector_manager.get_retriever(
                        search_kwargs={'k': Config.TOP_K_RESULTS}
                    )
                    st.session_state.rag_engine.create_qa_chain(retriever)
            else:
                st.warning("No content extracted from documents")
                
        except Exception as e:
            st.error(f" Error processing documents: {e}")

def qa_mode():
    """Q&A Mode Interface"""
    st.markdown('<div class="main-header"> Legal Q&A Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask questions about your legal documents</div>', unsafe_allow_html=True)
    
    if not st.session_state.vector_store_loaded:
        st.warning("⚠️ No documents indexed yet. Please upload documents from the sidebar.")
        return
    
    # Load RAG engine
    load_rag_engine()
    
    # Example questions
    with st.expander("💡 Example Questions"):
        st.markdown("""
        - What is the punishment for criminal breach of trust under IPC?
        - Explain Section 420 in simple words
        - What are the key provisions regarding contract termination?
        - Summarize the liability clause in the agreement
        """)
    
    # Chat interface
    for message in st.session_state.chat_history:
        display_message(message['role'], message['content'])
    
    # Query input
    query = st.chat_input("Ask a legal question...")
    
    if query:
        # Display user message
        display_message('user', query)
        st.session_state.chat_history.append({'role': 'user', 'content': query})
        
        # Get answer
        with st.spinner("🔍 Searching legal documents..."):
            result = st.session_state.rag_engine.query(query)
            
            answer = result['answer']
            sources = result['sources']
            
            # Format response
            response = f"{answer}\n\n{format_sources(sources)}"
            
            # Display assistant message
            display_message('assistant', response)
            st.session_state.chat_history.append({'role': 'assistant', 'content': response})

def summarization_mode():
    """Summarization Mode Interface"""
    st.markdown('<div class="main-header">📜Document Summarization</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Get concise summaries of legal documents</div>', unsafe_allow_html=True)
    
    # Load summarizer
    load_summarizer()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Text input option
        st.markdown("### Option 1: Paste Text")
        input_text = st.text_area(
            "Enter legal text to summarize",
            height=200,
            placeholder="Paste judgment, contract, or legal document text here..."
        )
    
    with col2:
        # File upload option
        st.markdown("### Option 2: Upload File")
        uploaded_file = st.file_uploader(
            "Upload document",
            type=['pdf', 'docx', 'txt']
        )
        
        summary_type = st.radio(
            "Summary Type",
            ["Quick Summary", "5-Point Legal Summary"]
        )
    
    if st.button("Generate Summary", type="primary"):
        text_to_summarize = ""
        
        # Get text from input or file
        if uploaded_file:
            with st.spinner("Reading document..."):
                temp_path = Path(tempfile.gettempdir()) / uploaded_file.name
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                doc_data = st.session_state.doc_processor.load_document(str(temp_path))
                text_to_summarize = doc_data['text']
        elif input_text:
            text_to_summarize = input_text
        else:
            st.warning("Please provide text or upload a document")
            return
        
        if text_to_summarize:
            with st.spinner(" Generating summary..."):
                if summary_type == "Quick Summary":
                    summary = st.session_state.summarizer.summarize_text(text_to_summarize)
                    st.markdown("### 📝 Summary")
                    st.success(summary)
                else:
                    summary_dict = st.session_state.summarizer.summarize_legal_judgment(text_to_summarize)
                    
                    st.markdown("### Legal Summary (5 Points)")
                    
                    st.markdown("** Key Facts**")
                    st.info(summary_dict['key_facts'])
                    
                    st.markdown("** Decision Summary**")
                    st.info(summary_dict['decision'])
                    
                    st.markdown("** Legal Basis**")
                    st.info(summary_dict['legal_basis'])
                    
                    st.markdown("**Precedent Used**")
                    st.info(summary_dict['precedent'])
                    
                    st.markdown("** Verdict**")
                    st.info(summary_dict['verdict'])

def comparison_mode():
    """Clause Comparison Mode Interface"""
    st.markdown('<div class="main-header"> Clause Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Compare legal documents and clauses semantically</div>', unsafe_allow_html=True)
    
    # Load comparator
    load_comparator()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("###  Document/Clause A")
        text1 = st.text_area(
            "Enter first text",
            height=200,
            placeholder="Paste first clause or document..."
        )
        name1 = st.text_input("Name (optional)", value="Document A", key="name1")
    
    with col2:
        st.markdown("###  Document/Clause B")
        text2 = st.text_area(
            "Enter second text",
            height=200,
            placeholder="Paste second clause or document..."
        )
        name2 = st.text_input("Name (optional)", value="Document B", key="name2")
    
    if st.button(" Compare", type="primary"):
        if not text1 or not text2:
            st.warning("Please provide both texts to compare")
            return
        
        with st.spinner(" Analyzing semantic similarity..."):
            comparison = st.session_state.comparator.compare_clauses(
                text1, text2, name1, name2
            )
            
            # Display results
            st.markdown("---")
            st.markdown("###  Comparison Results")
            
            # Similarity score
            score = comparison['similarity_score']
            emoji = get_similarity_emoji(score)
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Similarity Score", f"{emoji} {comparison['similarity_percentage']}")
            with col_b:
                st.metric("Interpretation", comparison['interpretation'])
            with col_c:
                st.metric("Documents", f"{name1}  {name2}")
            
            # Legal implications
            st.markdown("### ⚖️ Legal Implications")
            st.info(comparison['legal_implications'])
            
            # Key differences
            if comparison['differences']:
                st.markdown("###  Key Differences")
                for i, diff in enumerate(comparison['differences'][:5], 1):
                    with st.expander(f"Difference {i} - {diff['location']}"):
                        st.markdown(f"**Sentence:** {diff['sentence']}")
                        st.markdown(f"**Similarity to other text:** {diff['similarity_to_other']:.2%}")

def voice_mode():
    """Voice Mode Interface"""
    st.markdown('<div class="main-header"> Voice Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Speak your queries and hear responses</div>', unsafe_allow_html=True)
    
    if not st.session_state.vector_store_loaded:
        st.warning(" No documents indexed yet. Please upload documents from the sidebar.")
        return
    
    # Load components
    load_rag_engine()
    load_voice_handler()
    
    st.markdown("### 🎤 Record Your Question")
    st.info("Upload an audio file with your legal question (MP3, WAV, M4A)")
    
    audio_file = st.file_uploader(
        "Upload audio query",
        type=['mp3', 'wav', 'm4a', 'ogg']
    )
    
    if audio_file and st.button(" Process Voice Query", type="primary"):
        with st.spinner(" Processing voice query..."):
            # Save audio temporarily
            temp_audio = Path(tempfile.gettempdir()) / audio_file.name
            with open(temp_audio, 'wb') as f:
                f.write(audio_file.getbuffer())
            
            # Transcribe
            st.markdown("####  Transcription")
            transcription = st.session_state.voice_handler.transcribe_audio(str(temp_audio))
            
            if transcription['success']:
                query_text = transcription['text']
                st.success(f"**You asked:** {query_text}")
                
                # Get answer
                st.markdown("####  Answer")
                result = st.session_state.rag_engine.query(query_text)
                answer = result['answer']
                st.info(answer)
                
                # Convert to speech
                st.markdown("####  Audio Response")
                audio_path = st.session_state.voice_handler.text_to_speech(answer)
                
                with open(audio_path, 'rb') as audio_out:
                    st.audio(audio_out.read(), format='audio/mp3')
                
                # Show sources
                if result['sources']:
                    with st.expander(" Sources"):
                        st.markdown(format_sources(result['sources']))
            else:
                st.error(f" Transcription failed: {transcription['error']}")

def main():
    """Main application"""
    # Initialize
    init_session_state()
    
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 1rem; background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%); border-radius: 0.5rem; margin-bottom: 2rem;'>
        <h1 style='color: white; margin: 0;'> Legal AI Chat Assistant</h1>
        <p style='color: #E0E7FF; margin: 0.5rem 0 0 0;'>Advanced RAG System for Legal Document Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    sidebar()
    
    # Load components on first run
    if not st.session_state.initialized:
        if not load_components():
            st.stop()
    
    # Route to appropriate mode
    if st.session_state.current_mode == "Q&A":
        qa_mode()
    elif st.session_state.current_mode == "Summarization":
        summarization_mode()
    elif st.session_state.current_mode == "Clause Comparison":
        comparison_mode()
    elif st.session_state.current_mode == "Voice Mode":
        voice_mode()

if __name__ == "__main__":
    main()
