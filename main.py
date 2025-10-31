import streamlit as st
import os
import tempfile
from core.loader import DocumentProcessor
from core.chat import RAGChatEngine
from utils.helpers import setup_environment
import sys

def initialize_session_state():
    if 'chat_engine' not in st.session_state:
        st.session_state.chat_engine = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False

def main():
    st.set_page_config(
        page_title="DocuMind - Chat with Your Documents",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 DocuMind - Intelligent Document Assistant")
    st.markdown("Upload your documents and chat with them using AI-powered understanding")
    
    initialize_session_state()
    setup_environment()
    
    with st.sidebar:
        st.header("📄 Document Upload")
        st.markdown("Supported formats: PDF, Word (.docx), Excel (.xlsx)")
        
        uploaded_files = st.file_uploader(
            "Choose documents",
            type=['pdf', 'docx', 'xlsx'],
            accept_multiple_files=True,
            help="You can upload multiple documents at once"
        )
        
        process_button = st.button("Process Documents", type="primary")
        
        if process_button and uploaded_files:
            with st.spinner("Processing documents... This may take a few moments."):
                try:
                    temp_files = []
                    for uploaded_file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            temp_files.append(tmp_file.name)
                    
                    processor = DocumentProcessor()
                    documents = processor.process_documents(temp_files)
                    
                    if documents:
                        st.session_state.chat_engine = RAGChatEngine()
                        st.session_state.chat_engine.initialize_vector_store(documents)
                        st.session_state.documents_processed = True
                        st.session_state.messages = []
                        st.success(f"✅ Successfully processed {len(documents)} document chunks!")
                    else:
                        st.error("❌ No text content could be extracted from the documents.")
                        
                    for temp_file in temp_files:
                        os.unlink(temp_file)
                        
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")
        
        if st.session_state.documents_processed:
            st.success("✅ Documents ready for questioning!")
            if st.button("Clear Chat History"):
                st.session_state.messages = []
                st.rerun()
    
    st.header("💬 Chat Interface")
    
    if not st.session_state.documents_processed:
        st.info("👆 Please upload and process your documents in the sidebar to start chatting.")
        return
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chat_engine.ask_question(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.markdown(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()