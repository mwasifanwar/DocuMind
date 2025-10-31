import os
from typing import List, Optional
from langchain.schema import Document
from langchain.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
    
    def process_documents(self, file_paths: List[str]) -> Optional[List[Document]]:
        all_documents = []
        
        for file_path in file_paths:
            try:
                documents = self._load_single_document(file_path)
                if documents:
                    chunked_documents = self.text_splitter.split_documents(documents)
                    all_documents.extend(chunked_documents)
                    print(f"Processed {file_path}: {len(chunked_documents)} chunks")
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
        
        return all_documents if all_documents else None
    
    def _load_single_document(self, file_path: str) -> Optional[List[Document]]:
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == '.pdf':
                return self._load_pdf_document(file_path)
            elif file_extension == '.docx':
                return self._load_word_document(file_path)
            elif file_extension == '.xlsx':
                return self._load_excel_document(file_path)
            else:
                print(f"Unsupported file format: {file_extension}")
                return None
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return None
    
    def _load_pdf_document(self, file_path: str) -> List[Document]:
        loader = PyPDFLoader(file_path)
        return loader.load()
    
    def _load_word_document(self, file_path: str) -> List[Document]:
        loader = UnstructuredWordDocumentLoader(file_path)
        return loader.load()
    
    def _load_excel_document(self, file_path: str) -> List[Document]:
        loader = UnstructuredExcelLoader(file_path)
        return loader.load()

def test_loader():
    processor = DocumentProcessor()
    test_files = ["test.pdf", "test.docx", "test.xlsx"]
    existing_files = [f for f in test_files if os.path.exists(f)]
    
    if existing_files:
        documents = processor.process_documents(existing_files)
        if documents:
            print(f"Successfully processed {len(documents)} document chunks")
            return documents
    else:
        print("No test files found for testing")
    return None

if __name__ == "__main__":
    test_loader()