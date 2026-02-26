"""Domain-organized models for the Document MCP system.

Modern modular architecture with domain-driven design:
- analysis: Analytics, statistics, and semantic search models
- content: Document and chapter content models
- context: Context management and memory models
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
from .context import ExportStatus
from .context import ImportStatus
from .context import MemoryEntry
from .context import SessionMetadata
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
from .version_control import CommitInfo
from .version_control import VersionComparisonResult
from .version_control import VersionDiff
from .version_control import VersionHistory

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
    # context
    "ExportStatus",
    "ImportStatus",
    "MemoryEntry",
    "SessionMetadata",
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
    # version_control
    "CommitInfo",
    "VersionComparisonResult",
    "VersionDiff",
    "VersionHistory",
]
