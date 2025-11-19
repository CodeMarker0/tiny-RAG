"""
This module provides a flexible framework for loading data from various local file types,
specifically focusing on PDF and TXT documents. It is designed to be easily extensible
for supporting additional file formats in the future.

The core component of this module is the `LocalFileLoader`, a class that encapsulates
the logic for reading and processing files. It leverages specialized helper classes
for handling the intricacies of each supported file type, ensuring that the content
and metadata are extracted accurately.

Key Features:
- Support for PDF (.pdf) and plain text (.txt) files.
- Automatic detection of file types based on their extensions.
- Extraction of both content and basic metadata (e.g., file path, type).
- A clear and modular design that simplifies the process of adding new file loaders.
- Comprehensive error handling to gracefully manage issues like file not found or
  unsupported formats.

The module relies on external libraries such as `pypdf` for PDF processing, and it
integrates with the `loguru` library for structured and informative logging.
"""

import os
from typing import List, Dict, Any, Optional
from loguru import logger
from pypdf import PdfReader
import chardet


class LocalFileLoader:
    """
    A versatile loader for reading documents from local files.

    This class is designed to handle various file formats by delegating the processing
    to specialized handler methods. It currently supports PDF and plain text files,
    and it can be extended to accommodate other document types as needed.
    """

    def __init__(self, file_paths: List[str]):
        """
        Initializes the LocalFileLoader with a list of file paths.

        Args:
            file_paths: A list of strings, where each string is a path to a file
                        that needs to be loaded.
        """
        self.file_paths = file_paths

    def load(self) -> List[Dict[str, Any]]:
        """
        Loads and processes all the files specified during initialization.

        This method iterates through the list of file paths, determines the appropriate
        handler for each file based on its extension, and then uses that handler to
        extract the content and metadata.

        Returns:
            A list of dictionaries, where each dictionary represents a document.
            Each dictionary contains the document's content and metadata.
        """
        documents = []
        for file_path in self.file_paths:
            try:
                if not os.path.exists(file_path):
                    logger.warning(f"File not found: {file_path}")
                    continue

                file_extension = os.path.splitext(file_path)[1].lower()

                if file_extension == '.pdf':
                    doc = self._load_pdf(file_path)
                elif file_extension == '.txt':
                    doc = self._load_txt(file_path)
                else:
                    logger.warning(f"Unsupported file type: {file_extension}")
                    continue

                if doc:
                    documents.append(doc)

            except Exception as e:
                logger.error(f"Failed to load file {file_path}: {e}")

        return documents

    def _load_pdf(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Loads a PDF file, extracts its text content and metadata.

        Args:
            file_path: The path to the PDF file.

        Returns:
            A dictionary containing the content and metadata of the PDF,
            or None if the file cannot be processed.
        """
        try:
            reader = PdfReader(file_path)
            content = ""
            for page in reader.pages:
                content += page.extract_text() or ""

            metadata = {
                'source': file_path,
                'file_type': 'pdf',
                'num_pages': len(reader.pages)
            }
            return {'content': content, 'metadata': metadata}

        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {e}")
            return None

    def _load_txt(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Loads a plain text file, extracts its content and metadata.

        This method attempts to detect the file's encoding to handle various
        text formats correctly.

        Args:
            file_path: The path to the text file.

        Returns:
            A dictionary containing the content and metadata of the text file,
            or None if the file cannot be processed.
        """
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding'] if result['encoding'] else 'utf-8'

            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()

            metadata = {
                'source': file_path,
                'file_type': 'txt',
                'encoding': encoding
            }
            return {'content': content, 'metadata': metadata}

        except Exception as e:
            logger.error(f"Error reading TXT {file_path}: {e}")
            return None


if __name__ == "__main__":
    # Example usage of the LocalFileLoader
    # Create dummy files for demonstration
    if not os.path.exists("sample_data"):
        os.makedirs("sample_data")

    with open("sample_data/sample.txt", "w", encoding="utf-8") as f:
        f.write("This is a sample text file for testing the LocalFileLoader.")

    # To test PDF loading, you would need a sample PDF file.
    # For this example, we will only use the TXT file.
    # You can add a PDF file named "sample.pdf" in the "sample_data" directory.

    file_paths = ["sample_data/sample.txt"]
    # To test with a PDF, add its path here:
    # file_paths.append("sample_data/sample.pdf")

    loader = LocalFileLoader(file_paths)
    documents = loader.load()

    for doc in documents:
        print(f"Loaded document: {doc['metadata']['source']}")
        print(f"Content snippet: {doc['content'][:100]}...")
        print("-" * 20)