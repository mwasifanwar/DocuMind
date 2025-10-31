import os
import streamlit as st
from dotenv import load_dotenv

def setup_environment():
    load_dotenv()
    
    if 'OPENAI_API_KEY' not in os.environ:
        openai_key = st.sidebar.text_input(
            "Enter your OpenAI API Key:",
            type="password",
            help="Get your API key from https://platform.openai.com/account/api-keys"
        )
        if openai_key:
            os.environ['OPENAI_API_KEY'] = openai_key
            st.sidebar.success("✅ API Key saved for this session")
        else:
            st.sidebar.warning("🔑 Please enter your OpenAI API Key to continue")
            return False
    return True

def validate_file_type(filename: str) -> bool:
    allowed_extensions = {'.pdf', '.docx', '.xlsx'}
    file_extension = os.path.splitext(filename)[1].lower()
    return file_extension in allowed_extensions

def format_file_size(size_in_bytes: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024.0
    return f"{size_in_bytes:.2f} TB"

def cleanup_temp_files(file_paths: list):
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting temp file {file_path}: {str(e)}")

def get_supported_formats() -> list:
    return [
        {"format": "PDF", "extensions": ".pdf", "description": "Portable Document Format"},
        {"format": "Word", "extensions": ".docx", "description": "Microsoft Word Document"},
        {"format": "Excel", "extensions": ".xlsx", "description": "Microsoft Excel Spreadsheet"}
    ]