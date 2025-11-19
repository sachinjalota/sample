# src/utility/error_messages.py - NEW FILE

"""User-friendly error message mappings"""

ERROR_MESSAGES = {
    "extraction": {
        "timeout": "The file took too long to process. Please try with a smaller file or contact support.",
        "empty_content": "No readable text was found in the file. Please ensure the PDF contains selectable text, not just images.",
        "corrupted": "The file appears to be corrupted or in an unsupported format. Please check the file and try again.",
        "password_protected": "The file is password-protected. Please remove the password and upload again.",
    },
    "chunking": {
        "timeout": "Text processing took too long. The file may be too large or complex.",
        "failed": "Failed to split the document into chunks. This may be due to unusual formatting.",
    },
    "embedding": {
        "timeout": "Embedding generation timed out. The document may be too large.",
        "model_error": "Failed to generate embeddings. Please try again later or contact support.",
        "quota_exceeded": "Embedding service quota exceeded. Please try again later.",
    },
    "indexing": {
        "timeout": "Database indexing took too long. Please contact support.",
        "connection_error": "Failed to connect to the vector store. Please try again.",
        "duplicate": "A file with this name already exists. Use override=true to replace it.",
        "rollback": "Indexing failed and changes were rolled back. Your vector store remains unchanged.",
    }
}

def get_user_friendly_error(stage: str, error_type: str, fallback: str = None) -> str:
    """
    Get user-friendly error message
    
    Args:
        stage: Processing stage (extraction, chunking, embedding, indexing)
        error_type: Specific error type
        fallback: Original error message if no mapping exists
    
    Returns:
        User-friendly error message
    """
    messages = ERROR_MESSAGES.get(stage, {})
    return messages.get(error_type, fallback or f"An error occurred during {stage}. Please try again or contact support.")
