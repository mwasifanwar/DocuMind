import os
from typing import List, Optional
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGChatEngine:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self.vector_store = None
        self.qa_chain = None
        self.embeddings = None
        self.llm = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='result'
        )
        
    def initialize_vector_store(self, documents: List[Document], persist_directory: str = "./chroma_db"):
        try:
            self.embeddings = OpenAIEmbeddings()
            
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=persist_directory
            )
            
            self.vector_store.persist()
            
            self.llm = ChatOpenAI(
                model_name=self.model_name,
                temperature=0.1,
                streaming=False
            )
            
            prompt_template = """You are an intelligent document assistant called DocuMind. Use the following pieces of context to answer the user's question. 
            If you don't know the answer based on the provided context, just say you don't know based on the documents provided. Don't try to make up an answer.

            Context: {context}

            Question: {question}

            Provide a comprehensive answer based strictly on the document context. If the context doesn't contain relevant information, respond: "I couldn't find specific information about this in the uploaded documents."

            Helpful Answer:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4}
                ),
                chain_type_kwargs={"prompt": PROMPT, "memory": self.memory},
                return_source_documents=True,
                output_key='result'
            )
            
            logger.info("Vector store and QA chain initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise
    
    def ask_question(self, question: str) -> str:
        if not self.qa_chain:
            return "Please process documents first before asking questions."
        
        try:
            result = self.qa_chain({"query": question})
            answer = result['result']
            
            source_docs = result.get('source_documents', [])
            if source_docs:
                sources = list(set([os.path.basename(doc.metadata.get('source', 'Unknown')) for doc in source_docs]))
                if sources and sources[0]:
                    answer += f"\n\n📚 *Sources: {', '.join(sources)}*"
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"I encountered an error while processing your question: {str(e)}"
    
    def get_chat_history(self) -> List[str]:
        if hasattr(self, 'memory') and self.memory:
            return self.memory.load_memory_variables({}).get('chat_history', [])
        return []
    
    def clear_memory(self):
        if hasattr(self, 'memory') and self.memory:
            self.memory.clear()

def test_chat_engine():
    from core.loader import DocumentProcessor
    import tempfile
    
    print("Testing Chat Engine...")
    processor = DocumentProcessor()
    chat_engine = RAGChatEngine()
    
    test_docs = [
        Document(
            page_content="This is a test document about artificial intelligence. AI is transforming many industries.",
            metadata={"source": "test_doc.txt"}
        )
    ]
    
    try:
        chat_engine.initialize_vector_store(test_docs)
        response = chat_engine.ask_question("What is AI?")
        print(f"Test Response: {response}")
        return True
    except Exception as e:
        print(f"Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_chat_engine()