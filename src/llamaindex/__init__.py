"""
LlamaIndex Cloud Services Integration for Financial Document Extraction

This package provides tools for extracting structured data from financial documents
using LlamaExtract with full citation and reasoning capabilities.
"""

from .extractor import (
    extract_documents,
    IncomeStatement,
    get_extraction_agent
)

__all__ = [
    'extract_documents',
    'IncomeStatement',
    'get_extraction_agent'
]

