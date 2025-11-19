"""
RAG System Main Entry Point
Provides a command-line interface and the complete workflow for the RAG system.
"""
import argparse
import os
import sys
from typing import Optional, List

from src import (
    settings,
    LocalFileLoader,
    DocumentProcessor,
    VectorStoreManager,
    RAGSystem
)
from src.utils import setup_logger, print_banner
from loguru import logger


class RAGApplication:
    """Main class for the RAG application."""

    def __init__(self):
        """Initializes the application."""
        # Set up logging
        setup_logger(
            log_level=settings.log_level,
            log_file=settings.log_file
        )

        # Print banner
        print_banner()

        # Initialize components
        self.doc_processor: Optional[DocumentProcessor] = None
        self.vector_manager: Optional[VectorStoreManager] = None
        self.rag_system: Optional[RAGSystem] = None

    def initialize_components(self):
        """Initializes all system components."""
        logger.info("Initializing system components...")

        # Initialize document processor
        self.doc_processor = DocumentProcessor(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )

        # Initialize vector store manager
        self.vector_manager = VectorStoreManager(
            embedding_model_name=settings.embedding_model_name,
            persist_directory=settings.faiss_persist_directory,
            index_name=settings.index_name
        )

        # Try to load an existing vector database
        if not self.vector_manager.load_vectorstore():
            logger.warning("Existing vector database not found. You need to build the knowledge base first.")

        # Initialize RAG system
        self.rag_system = RAGSystem(
            vector_store_manager=self.vector_manager,
            llm_model_name=settings.llm_model_name,
            ollama_base_url=settings.ollama_base_url,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            retrieval_top_k=settings.retrieval_top_k
        )

        logger.info("System components initialized successfully.")

    def build_knowledge_base(self, file_paths: List[str], force_rebuild: bool = False):
        """
        Builds the knowledge base from local files.

        Args:
            file_paths: A list of paths to the files to be loaded.
            force_rebuild: Whether to force a rebuild of the knowledge base.
        """
        try:
            logger.info("=" * 50)
            logger.info("Starting to build the knowledge base...")
            logger.info("=" * 50)

            # Step 1: Load documents from local files
            logger.info(f"Step 1/4: Loading documents from {len(file_paths)} files...")
            data_loader = LocalFileLoader(file_paths=file_paths)
            raw_documents = data_loader.load()

            if not raw_documents:
                logger.error("No documents were loaded. Cannot build the knowledge base.")
                return False

            # Step 2: Process documents
            logger.info(f"Step 2/4: Processing {len(raw_documents)} documents...")
            processed_docs = self.doc_processor.process_documents(raw_documents)

            # Step 3: Split documents
            logger.info("Step 3/4: Splitting documents into chunks...")
            chunks = self.doc_processor.split_documents(processed_docs)

            # Step 4: Create vector database
            logger.info(f"Step 4/4: Creating vector database with {len(chunks)} chunks...")
            self.vector_manager.create_vectorstore(
                documents=chunks,
                replace_existing=force_rebuild
            )

            # Re-initialize the RAG system's QA chain
            self.rag_system.setup_qa_chain()

            logger.info("=" * 50)
            logger.info("Knowledge base built successfully!")
            logger.info("=" * 50)

            # Print statistics
            stats = self.vector_manager.get_statistics()
            logger.info(f"Knowledge base statistics: {stats}")

            return True

        except Exception as e:
            logger.error(f"Failed to build knowledge base: {e}")
            return False

    def interactive_query(self):
        """Enters interactive query mode."""
        logger.info("Entering interactive query mode (type 'quit' or 'exit' to leave).")
        print("\n" + "=" * 50)
        print("RAG system is ready. Please enter your question.")
        print("=" * 50 + "\n")

        while True:
            try:
                # Get user input
                question = input("\n❓ Your question: ").strip()

                if not question:
                    continue

                if question.lower() in ['quit', 'exit', 'q']:
                    logger.info("Exiting interactive query mode.")
                    break

                # Perform query
                print("\n🔍 Searching...\n")
                result = self.rag_system.query(question)

                # Display results
                print("=" * 50)
                print(f"📝 Answer:\n{result['answer']}")
                print("=" * 50)

                # Display source documents
                if result['sources']:
                    print(f"\n📚 Source documents ({len(result['sources'])}):")
                    for source in result['sources']:
                        print(f"\n  [{source['index']}] {source['title']}")
                        print(f"      Type: {source.get('type', 'N/A')} | Date: {source.get('date', 'N/A')}")
                        print(f"      Snippet: {source['content'][:200]}...")

            except KeyboardInterrupt:
                logger.info("\nInterrupt signal detected. Exiting...")
                break
            except Exception as e:
                logger.error(f"An error occurred during query: {e}")
                print(f"\n❌ Query failed: {e}\n")

    def single_query(self, question: str):
        """
        Performs a single query.

        Args:
            question: The question to ask.
        """
        logger.info(f"Performing a single query: {question}")
        result = self.rag_system.query(question)

        print("\n" + "=" * 50)
        print(f"Question: {result['question']}")
        print("=" * 50)
        print(f"Answer:\n{result['answer']}")
        print("=" * 50)

        if result['sources']:
            print(f"\nSource documents ({len(result['sources'])}):")
            for source in result['sources']:
                print(f"\n  [{source['index']}] {source['title']}")
                print(f"      {source['content'][:150]}...")

    def show_statistics(self):
        """Displays system statistics."""
        print("\n" + "=" * 50)
        print("System Statistics")
        print("=" * 50)

        stats = self.vector_manager.get_statistics()
        for key, value in stats.items():
            print(f"{key}: {value}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="A RAG system for querying local documents.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--build',
        nargs='*',
        metavar='FILE',
        help='Build or update the knowledge base. If no files are specified, it processes all files in the data/raw directory.'
    )

    parser.add_argument(
        '--rebuild',
        action='store_true',
        help='Force a rebuild of the knowledge base. Use with --build.'
    )

    parser.add_argument(
        '--query',
        type=str,
        help='Perform a single query.'
    )

    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Enter interactive query mode.'
    )

    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show system statistics.'
    )

    args = parser.parse_args()

    # Create an instance of the application
    app = RAGApplication()

    try:
        # Initialize components
        app.initialize_components()

        # Execute the appropriate action
        if args.build is not None:
            files_to_build = args.build
            # If --build is specified but no files are provided,
            # default to all files in the 'data/raw' directory.
            if not files_to_build:
                raw_data_dir = 'data/raw'
                logger.info(f"No files specified. Searching for documents in '{raw_data_dir}'...")
                if not os.path.isdir(raw_data_dir):
                    logger.error(f"Directory '{raw_data_dir}' not found. Please create it and add your documents.")
                    sys.exit(1)

                files_to_build = [os.path.join(raw_data_dir, f) for f in os.listdir(raw_data_dir)
                                  if os.path.isfile(os.path.join(raw_data_dir, f))]

                # Filter out hidden files like .gitkeep
                files_to_build = [f for f in files_to_build if not os.path.basename(f).startswith('.')]

                if not files_to_build:
                    logger.warning(f"No documents found in '{raw_data_dir}'. Knowledge base will not be built.")
                    return

            app.build_knowledge_base(file_paths=files_to_build, force_rebuild=args.rebuild)

        elif args.query:
            app.single_query(args.query)

        elif args.stats:
            app.show_statistics()

        elif args.interactive:
            app.interactive_query()

        else:
            # Default to interactive mode
            parser.print_help()
            print("\nTip: Use --interactive to enter interactive query mode.")

    except KeyboardInterrupt:
        logger.info("\nProgram interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"An error occurred during program execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
