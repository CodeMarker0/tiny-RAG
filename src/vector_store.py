"""
Vector Store Module
Uses the FAISS vector database to store and retrieve document vectors.
"""
from typing import List, Optional, Dict, Any
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from loguru import logger
import os


class VectorStoreManager:
    """Manages the vector store using FAISS."""

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_directory: str = "./data/vectorstore",
        index_name: str = "faiss_index"
    ):
        """
        Initializes the VectorStoreManager.

        Args:
            embedding_model_name: The name of the embedding model to use.
            persist_directory: The directory to persist the vector store to.
            index_name: The name of the index file (without extension).
        """
        self.embedding_model_name = embedding_model_name
        self.persist_directory = persist_directory
        self.index_name = index_name

        # FAISS index file paths
        self.index_file = os.path.join(persist_directory, f"{index_name}.faiss")
        self.pkl_file = os.path.join(persist_directory, f"{index_name}.pkl")

        # Ensure the directory exists
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize the embedding model
        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.embeddings = self._initialize_embeddings()

        # Initialize the vector store
        self.vectorstore = None

    def _initialize_embeddings(self) -> Embeddings:
        """
        Initializes the embedding model.

        Returns:
            An Embeddings object.
        """
        try:
            # Use HuggingFace Embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={'device': 'cpu'},  # Can be changed to 'cuda' to use GPU
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("Embedding model loaded successfully.")
            return embeddings

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def create_vectorstore(
        self,
        documents: List[Document],
        replace_existing: bool = False
    ) -> FAISS:
        """
        Creates a vector database.

        Args:
            documents: A list of documents.
            replace_existing: Whether to replace the existing database.

        Returns:
            A FAISS vector database instance.
        """
        try:
            # If replacing the existing database, delete it first
            if replace_existing:
                self._delete_index_files()

            logger.info(f"Creating vector database with {len(documents)} documents.")

            # Create the FAISS vector database
            self.vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )

            # Save to disk
            self._save_vectorstore()
            logger.info("Vector database created successfully.")

            return self.vectorstore

        except Exception as e:
            logger.error(f"Failed to create vector database: {e}")
            raise

    def load_vectorstore(self) -> Optional[FAISS]:
        """
        Loads an existing vector database.

        Returns:
            A FAISS vector database instance, or None if it doesn't exist.
        """
        try:
            if not os.path.exists(self.index_file) or not os.path.exists(self.pkl_file):
                logger.warning(f"Vector database not found in: {self.persist_directory}")
                return None

            logger.info(f"Loading vector database from: {self.persist_directory}")

            # Load from disk
            self.vectorstore = FAISS.load_local(
                folder_path=self.persist_directory,
                embeddings=self.embeddings,
                index_name=self.index_name,
                allow_dangerous_deserialization=True  # Required by FAISS
            )

            logger.info("Vector database loaded successfully.")
            return self.vectorstore

        except Exception as e:
            logger.error(f"Failed to load vector database: {e}")
            return None

    def add_documents(
        self,
        documents: List[Document]
    ) -> List[str]:
        """
        Adds documents to an existing vector database.

        Args:
            documents: A list of documents.

        Returns:
            A list of added document IDs.
        """
        try:
            if self.vectorstore is None:
                logger.warning("Vector database is not initialized. Creating a new one.")
                self.create_vectorstore(documents)
                return []

            logger.info(f"Adding {len(documents)} documents to the vector database.")

            # Add documents
            ids = self.vectorstore.add_documents(documents)

            # Save to disk
            self._save_vectorstore()

            logger.info(f"Successfully added {len(ids)} documents.")
            return ids

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Performs a similarity search.

        Args:
            query: The query text.
            k: The number of results to return.
            filter_dict: A dictionary for metadata filtering.

        Returns:
            A list of similar documents.
        """
        try:
            if self.vectorstore is None:
                logger.error("Vector database is not initialized.")
                return []

            logger.info(f"Searching for: {query[:50]}...")

            results = self.vectorstore.similarity_search(query, k=k, filter=filter_dict)

            logger.info(f"Found {len(results)} relevant documents.")
            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[tuple]:
        """
        Performs a similarity search with scores.

        Args:
            query: The query text.
            k: The number of results to return.
            filter_dict: A dictionary for metadata filtering.

        Returns:
            A list of (document, similarity_score) tuples.
        """
        try:
            if self.vectorstore is None:
                logger.error("Vector database is not initialized.")
                return []

            logger.info(f"Searching with scores for: {query[:50]}...")

            results = self.vectorstore.similarity_search_with_score(query, k=k, filter=filter_dict)

            logger.info(f"Found {len(results)} relevant documents.")

            for i, (doc, score) in enumerate(results):
                logger.debug(f"Result {i+1}: Score={score:.4f}, Title={doc.metadata.get('source', 'N/A')}")

            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def _save_vectorstore(self):
        """Saves the vector database to disk."""
        try:
            if self.vectorstore is None:
                logger.warning("Vector database is empty. Skipping save.")
                return

            self.vectorstore.save_local(
                folder_path=self.persist_directory,
                index_name=self.index_name
            )
            logger.info(f"Vector database saved to: {self.persist_directory}")

        except Exception as e:
            logger.error(f"Failed to save vector database: {e}")
            raise

    def _delete_index_files(self):
        """Deletes the index files."""
        try:
            if os.path.exists(self.index_file):
                os.remove(self.index_file)
                logger.info(f"Deleted index file: {self.index_file}")

            if os.path.exists(self.pkl_file):
                os.remove(self.pkl_file)
                logger.info(f"Deleted pkl file: {self.pkl_file}")

        except Exception as e:
            logger.error(f"Failed to delete index files: {e}")

    def delete_collection(self):
        """Deletes the entire vector collection."""
        try:
            self._delete_index_files()
            self.vectorstore = None
            logger.info("Vector database has been deleted.")

        except Exception as e:
            logger.error(f"Failed to delete vector database: {e}")
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """
        Gets statistics about the vector database.

        Returns:
            A dictionary of statistics.
        """
        try:
            if self.vectorstore is None:
                return {"status": "not_initialized"}

            document_count = self.vectorstore.index.ntotal

            stats = {
                "status": "active",
                "index_name": self.index_name,
                "document_count": document_count,
                "persist_directory": self.persist_directory,
                "embedding_model": self.embedding_model_name,
                "index_file_exists": os.path.exists(self.index_file),
                "pkl_file_exists": os.path.exists(self.pkl_file)
            }

            return stats

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {"status": "error", "error": str(e)}
