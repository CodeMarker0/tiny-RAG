"""
RAG Query System Module
Integrates vector retrieval and LLM generation to implement Retrieval-Augmented Generation.
"""
from typing import List, Dict, Any, Optional
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from loguru import logger

from src.vector_store import VectorStoreManager


class RAGSystem:
    """Main class for the RAG system."""

    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        llm_model_name: str = "qwen3:4b",
        ollama_base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        retrieval_top_k: int = 5
    ):
        """
        Initializes the RAG system.

        Args:
            vector_store_manager: The vector store manager.
            llm_model_name: The name of the LLM model to use.
            ollama_base_url: The base URL for the Ollama service.
            temperature: The temperature for text generation.
            max_tokens: The maximum number of tokens to generate.
            retrieval_top_k: The number of documents to retrieve.
        """
        self.vector_store_manager = vector_store_manager
        self.llm_model_name = llm_model_name
        self.ollama_base_url = ollama_base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retrieval_top_k = retrieval_top_k

        # Initialize the LLM
        logger.info(f"Initializing LLM: {llm_model_name}")
        self.llm = self._initialize_llm()

        # Initialize the retrieval chain
        self.qa_chain = None

    def _initialize_llm(self):
        """
        Initializes the local LLM.

        Returns:
            An instance of the LLM.
        """
        try:
            # Use Ollama as the local LLM service
            llm = Ollama(
                model=self.llm_model_name,
                base_url=self.ollama_base_url,
                temperature=self.temperature,
                num_predict=self.max_tokens,
                callbacks=[StreamingStdOutCallbackHandler()]  # Stream the output
            )
            logger.info("LLM initialized successfully.")
            return llm

        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            logger.warning("Please ensure that Ollama is installed and running.")
            raise

    def setup_qa_chain(self):
        """
        Sets up the question-answering chain.

        Returns:
            A RetrievalQA chain.
        """
        try:
            if self.vector_store_manager.vectorstore is None:
                logger.error("Vector database is not initialized.")
                raise ValueError("Vector database is not initialized. Please load or create it first.")

            # Define the English prompt template
            prompt_template = """You are a professional assistant responsible for answering questions about local documents.

Use the following retrieved context to answer the question. If you don't know the answer, just say that you don't know. Don't try to make up an answer.
Provide a detailed and professional answer, and cite the specific document content whenever possible. Attention: Answer in the language of the input.

Context:
{context}

Question: {question}

Detailed Answer:"""

            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )

            # Create the retriever
            retriever = self.vector_store_manager.vectorstore.as_retriever(
                search_kwargs={"k": self.retrieval_top_k}
            )

            # Create the RetrievalQA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",  # Pass all retrieved documents to the LLM at once
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )

            logger.info("Question-answering chain set up successfully.")
            return self.qa_chain

        except Exception as e:
            logger.error(f"Failed to set up QA chain: {e}")
            raise

    def query(
        self,
        question: str,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Performs a RAG query.

        Args:
            question: The user's question.
            return_sources: Whether to return the source documents.

        Returns:
            A dictionary containing the answer and source documents.
        """
        try:
            if self.qa_chain is None:
                logger.warning("QA chain is not initialized. Initializing now...")
                self.setup_qa_chain()

            logger.info(f"Processing question: {question}")

            # Perform the query
            result = self.qa_chain({"query": question})

            # Build the response
            response = {
                "question": question,
                "answer": result.get("result", ""),
                "sources": []
            }

            # Add source document information
            if return_sources and "source_documents" in result:
                for i, doc in enumerate(result["source_documents"]):
                    source_info = {
                        "index": i + 1,
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "title": doc.metadata.get("source", "Unknown document"),
                        "type": doc.metadata.get("file_type", "unknown"),
                    }
                    response["sources"].append(source_info)

            logger.info("Query completed.")
            return response

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                "question": question,
                "answer": f"An error occurred during the query: {str(e)}",
                "sources": []
            }
