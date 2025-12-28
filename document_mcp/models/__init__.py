"""Domain-organized models for the Document MCP system.

Modern modular architecture with domain-driven design:
- analysis: Analytics, statistics, and semantic search models
- content: Document and chapter content models
- core: Base operation and status models
- documents: Document metadata and structure models
"""

from .analysis import ChapterEmbeddingManifest
from .analysis import EmbeddingCacheEntry
from .analysis import SemanticSearchResponse
from .analysis import SemanticSearchResult
from .analysis import StatisticsReport
from .content import ChapterContent
from .content import FullDocumentContent
from .content import PaginatedContent
from .content import PaginationInfo
from .core import ContentFreshnessStatus
from .core import ModificationHistory
from .core import ModificationHistoryEntry
from .core import OperationStatus
from .core import ParagraphDetail
from .documents import ChapterMetadata
from .documents import DocumentInfo
from .documents import DocumentSummary
from .documents import SnapshotInfo
from .documents import SnapshotsList
from .metadata import ChapterMetadataInput
from .metadata import EntityDataInput
from .metadata import MetadataFilterInput
from .metadata import MetadataListResponse
from .metadata import MetadataResponse
from .metadata import TimelineEventInput

__all__ = [
    # analysis
    "ChapterEmbeddingManifest",
    "EmbeddingCacheEntry",
    "SemanticSearchResponse",
    "SemanticSearchResult",
    "StatisticsReport",
    # content
    "ChapterContent",
    "FullDocumentContent",
    "PaginatedContent",
    "PaginationInfo",
    # core
    "ContentFreshnessStatus",
    "ModificationHistory",
    "ModificationHistoryEntry",
    "OperationStatus",
    "ParagraphDetail",
    # documents
    "ChapterMetadata",
    "DocumentInfo",
    "DocumentSummary",
    "SnapshotInfo",
    "SnapshotsList",
    # metadata
    "ChapterMetadataInput",
    "EntityDataInput",
    "MetadataFilterInput",
    "MetadataListResponse",
    "MetadataResponse",
    "TimelineEventInput",
]
