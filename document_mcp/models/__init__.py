"""Domain-organized models for the Document MCP system.

Modern modular architecture with domain-driven design:
- analysis: Analytics, statistics, and semantic search models
- content: Document and chapter content models
- core: Base operation and status models
- documents: Document metadata and structure models
"""

from .analysis import *
from .content import *
from .core import *
from .documents import *
