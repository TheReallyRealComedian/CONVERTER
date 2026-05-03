# services/pdf_extraction_service.py
# Backward-compatibility shim — actual implementation in pdf_extraction/service.py
from .pdf_extraction.service import PDFExtractionService

__all__ = ['PDFExtractionService']
