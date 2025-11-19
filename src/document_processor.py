"""
Document Processing and Chunking Module
Handles various document formats and performs intelligent chunking.
"""
from typing import List, Dict, Any, Optional
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)
from langchain.docstore.document import Document
from loguru import logger
import re


class DocumentProcessor:
    """Handles document processing tasks."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None
    ):
        """
        Initializes the DocumentProcessor.

        Args:
            chunk_size: The size of each chunk.
            chunk_overlap: The overlap between chunks.
            separators: A list of separators to use for splitting text.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Use English-friendly separators
        if separators is None:
            separators = [
                "\n\n",  # Paragraphs
                "\n",    # Newlines
                ".",     # Period
                "!",     # Exclamation mark
                "?",     # Question mark
                ";",     # Semicolon
                ",",     # Comma
                " ",     # Space
                "",      # Characters
            ]

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )

    def process_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Document]:
        """
        Processes a list of raw documents.

        Args:
            documents: A list of raw documents, where each document is a dictionary.

        Returns:
            A list of processed LangChain Document objects.
        """
        processed_docs = []

        for doc in documents:
            try:
                # Extract content and metadata
                content = doc.get('content', '')
                metadata = doc.get('metadata', {})

                if not content or not isinstance(content, str):
                    logger.warning(f"Skipping invalid document: {metadata.get('id', 'unknown')}")
                    continue

                # Clean the text
                cleaned_content = self._clean_text(content)

                # Create a LangChain Document object
                langchain_doc = Document(
                    page_content=cleaned_content,
                    metadata=metadata
                )

                processed_docs.append(langchain_doc)

            except Exception as e:
                logger.error(f"Failed to process document: {e}")
                continue

        logger.info(f"Successfully processed {len(processed_docs)} documents.")
        return processed_docs

    def split_documents(
        self,
        documents: List[Document]
    ) -> List[Document]:
        """
        Splits a list of documents into smaller chunks.

        Args:
            documents: A list of LangChain Document objects.

        Returns:
            A list of chunked Document objects.
        """
        try:
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Document splitting complete: {len(documents)} documents -> {len(chunks)} chunks.")

            # Add additional metadata to each chunk
            for i, chunk in enumerate(chunks):
                chunk.metadata['chunk_id'] = i
                chunk.metadata['chunk_size'] = len(chunk.page_content)

            return chunks

        except Exception as e:
            logger.error(f"Failed to split documents: {e}")
            raise

    def _clean_text(self, text: str) -> str:
        """
        Cleans the input text.

        Args:
            text: The original text.

        Returns:
            The cleaned text.
        """
        if not text:
            return ""

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove extra newlines
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Trim leading/trailing whitespace
        text = text.strip()

        return text
