"""
Configuration Management Module
Uses pydantic-settings for configuration management.
"""
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """System configuration class."""

    # Embedding model configuration
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="The name of the embedding model to use."
    )

    # LLM configuration
    llm_model_name: str = Field(
        default="qwen3:4b",
        description="The name of the local LLM model to use."
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="The base URL for the Ollama service."
    )

    # Vector database configuration (FAISS)
    faiss_persist_directory: str = Field(
        default="./data/vectorstore",
        description="The directory to persist the FAISS index to."
    )
    index_name: str = Field(
        default="faiss_index",
        description="The name of the FAISS index."
    )

    # Document processing configuration
    chunk_size: int = Field(
        default=500,
        description="The size of the chunks to split documents into."
    )
    chunk_overlap: int = Field(
        default=50,
        description="The overlap between chunks."
    )
    max_documents: int = Field(
        default=1000,
        description="The maximum number of documents to load."
    )

    # RAG retrieval configuration
    retrieval_top_k: int = Field(
        default=5,
        description="The number of top-k results to retrieve."
    )
    temperature: float = Field(
        default=0.7,
        description="The temperature for the LLM generation."
    )
    max_tokens: int = Field(
        default=2000,
        description="The maximum number of tokens for the LLM to generate."
    )

    # Logging configuration
    log_level: str = Field(
        default="INFO",
        description="The logging level."
    )
    log_file: str = Field(
        default="./logs/rag_system.log",
        description="The path to the log file."
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Create a global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Returns the settings instance."""
    return settings
