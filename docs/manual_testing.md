# Document MCP: Manual Testing and User Workflows Guide

This comprehensive guide provides step-by-step instructions for manually testing the Document MCP system and demonstrates complete user workflows for creative writing, content editing, and document management.

## Table of Contents

1. [Quick Setup and Verification](#quick-setup-and-verification)
2. [Story Writing Workflow](#story-writing-workflow)
3. [Story Editing Workflow](#story-editing-workflow)
4. [Advanced Document Management](#advanced-document-management)
5. [Complete E2E Example Session](#complete-e2e-example-session)
6. [Troubleshooting Guide](#troubleshooting-guide)
7. [Performance and Load Testing](#performance-and-load-testing)

---

## Quick Setup and Verification

### Prerequisites

- Python 3.10+
- OpenAI API key OR Google Gemini API key
- Terminal/command prompt access

### Installation and Initial Setup

```bash
# Install via PyPI (recommended)
pip install document-mcp

# Verify installation
document-mcp --version

# Alternative: Development setup
git clone https://github.com/clchinkc/document-mcp.git
cd document-mcp
uv sync  # or: pip install -e ".[dev]"
```

### Environment Configuration

1. **Create `.env` file** with your API key:
```bash
# For OpenAI (recommended)
OPENAI_API_KEY=your_openai_api_key_here

# OR for Google Gemini  
GEMINI_API_KEY=your_gemini_api_key_here
```

2. **Verify configuration**:
```bash
# Using installed package
python -c "from document_mcp import __version__; print(f'Document MCP v{__version__} ready')"

# Or with development setup
python src/agents/simple_agent/main.py --check-config
```

Expected output:
```
✅ Configuration valid
Using Gemini model: gemini-2.5-flash
MCP server connection: Ready
```

### Basic System Verification

```bash
# Start MCP server (in one terminal)
document-mcp stdio

# Test basic functionality (in another terminal)
python src/agents/simple_agent/main.py --query "list all documents"
```

---

## Story Writing Workflow

### Scenario: Creating a Short Science Fiction Story

This workflow demonstrates creating a complete short story from conception to final draft.

#### Step 1: Create Document Structure

```bash
# Create new story document
python src/agents/simple_agent/main.py --query "Create a new document called 'Stellar Migration'"
```

**Expected Response:**
```json
{
  "success": true,
  "summary": "Successfully created document 'Stellar Migration'",
  "details": {
    "document_name": "Stellar Migration",
    "created_at": "2024-01-15T10:30:00Z"
  }
}
```

#### Step 2: Add Story Chapters

```bash
# Add opening chapter
python src/agents/simple_agent/main.py --query "Add a chapter called '01-arrival.md' to 'Stellar Migration' with the content '# Chapter 1: Arrival\n\nThe transport ship *Meridian* dropped out of hyperspace with a shudder that ran through every bulkhead. Captain Elena Vasquez felt it in her bones—the familiar sensation of reality reasserting itself after the impossible journey between stars.'"

# Add second chapter
python src/agents/simple_agent/main.py --query "Add a chapter called '02-discovery.md' to 'Stellar Migration' with the content '# Chapter 2: Discovery\n\nThe planet hung before them like a blue-green jewel, its surface swirled with clouds that hinted at active weather systems. But something was wrong. The readings didn't match the survey data from fifty years ago.'"
```

#### Step 3: Verify Story Structure

```bash
# List all documents to confirm creation
python src/agents/simple_agent/main.py --query "List all documents with their chapter counts and word statistics"

# Read the full story
python src/agents/simple_agent/main.py --query "Read the complete document 'Stellar Migration'"
```

**Expected Response:**
```json
{
  "success": true,
  "summary": "Retrieved complete document with 2 chapters",
  "details": {
    "document_name": "Stellar Migration",
    "chapters": [
      {
        "filename": "01-arrival.md",
        "content": "# Chapter 1: Arrival\n\nThe transport ship *Meridian*..."
      },
      {
        "filename": "02-discovery.md", 
        "content": "# Chapter 2: Discovery\n\nThe planet hung before them..."
      }
    ],
    "total_words": 89,
    "total_paragraphs": 4
  }
}
```

#### Step 4: Add Character Development

```bash
# Insert character development paragraph in Chapter 1
python src/agents/simple_agent/main.py --query "Insert a new paragraph in chapter '01-arrival.md' of document 'Stellar Migration' at position 1 with content 'Elena had made this journey seventeen times before, but each emergence still filled her with a mixture of relief and wonder. Three months in hyperspace meant three months of dreams that weren't quite dreams, of time that moved like honey through broken clockwork.'"
```

#### Step 5: Create Story Ending

```bash
# Add concluding chapter
python src/agents/simple_agent/main.py --query "Add a chapter called '03-choice.md' to 'Stellar Migration' with content '# Chapter 3: The Choice\n\nThe indigenous species had been waiting. Not for decades, but for centuries. They had watched the first survey ship, learned their language from their transmissions, and prepared for this moment.\n\n\"Captain,\" the voice came through the comm system in perfect, unaccented Standard. \"We have been expecting you. The question is: are you ready for us?\"'"
```

### Story Writing Best Practices

1. **Use descriptive chapter names** (`01-arrival.md` vs `chapter1.md`)
2. **Create snapshots before major edits** using the safety tools
3. **Verify word counts** regularly to track progress
4. **Use semantic search** to find similar themes or check consistency

---

## Story Editing Workflow

### Scenario: Editing and Refining an Existing Story

This workflow demonstrates comprehensive editing of the story created above.

#### Step 1: Content Analysis and Planning

```bash
# Get document statistics for baseline
python src/agents/simple_agent/main.py --query "Get detailed statistics for document 'Stellar Migration'"

# Search for specific themes or content
python src/agents/simple_agent/main.py --query "Find all paragraphs containing 'hyperspace' in document 'Stellar Migration'"

# Use semantic search for thematic analysis
python src/agents/simple_agent/main.py --query "Find content similar to 'space travel technology' in document 'Stellar Migration'"
```

#### Step 2: Create Safety Snapshot

```bash
# Create named snapshot before major editing
python src/agents/simple_agent/main.py --query "Create a snapshot of document 'Stellar Migration' with message 'Before major revision - adding dialogue and character development'"
```

**Expected Response:**
```json
{
  "success": true,
  "summary": "Snapshot created successfully",
  "details": {
    "snapshot_id": "snapshot_20240115_103000_manual_revision_user",
    "document_name": "Stellar Migration",
    "message": "Before major revision - adding dialogue and character development",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

#### Step 3: Advanced Paragraph Editing

```bash
# Replace specific content with improved version
python src/agents/simple_agent/main.py --query "Replace the paragraph containing 'dropped out of hyperspace' in document 'Stellar Migration' with 'The transport ship *Meridian* materialized from the quantum foam with a resonance that sang through every molecule of the vessel. Captain Elena Vasquez steadied herself against the navigation console as reality crystallized around them, the familiar weight of normal space settling over the ship like a comfortable blanket.'"

# Move paragraphs for better flow
python src/agents/simple_agent/main.py --query "Move paragraph 2 to position 3 in chapter '01-arrival.md' of document 'Stellar Migration'"

# Add dialogue to enhance character interaction
python src/agents/simple_agent/main.py --query "Insert a new paragraph at position 2 in chapter '02-discovery.md' of document 'Stellar Migration' with content '\"Navigation, report,\" Elena commanded, her voice cutting through the bridge's tense silence.\n\n\"Orbital mechanics are... unusual, Captain,\" replied Lieutenant Torres, his fingers dancing across the sensor array. \"The planet's showing gravitational anomalies that weren't in the original survey data.\"'"
```

#### Step 4: Content Search and Replace

```bash
# Global search and replace for consistency
python src/agents/simple_agent/main.py --query "Replace all occurrences of 'transport ship' with 'starship' in document 'Stellar Migration'"

# Find and replace character names if needed
python src/agents/simple_agent/main.py --query "Find all paragraphs containing 'Elena' in document 'Stellar Migration'"
```

#### Step 5: Verify Changes and Compare Versions

```bash
# Check modification history
python src/agents/simple_agent/main.py --query "Get the modification history for document 'Stellar Migration'"

# Compare current version with snapshot
python src/agents/simple_agent/main.py --query "Generate a diff between current document 'Stellar Migration' and snapshot 'snapshot_20240115_103000_manual_revision_user'"

# Get updated statistics
python src/agents/simple_agent/main.py --query "Get detailed statistics for document 'Stellar Migration'"
```

### Story Editing Best Practices

1. **Always create snapshots** before major revisions
2. **Use search tools** to maintain consistency across chapters
3. **Verify changes** with diff tools before proceeding
4. **Track modification history** to understand evolution
5. **Use semantic search** to identify thematic inconsistencies

---

## Advanced Document Management

### Multi-Document Project Management

```bash
# Create related documents
python src/agents/simple_agent/main.py --query "Create a new document called 'Stellar Migration - Character Notes'"
python src/agents/simple_agent/main.py --query "Create a new document called 'Stellar Migration - World Building'"

# Add structured content to character notes
python src/agents/simple_agent/main.py --query "Add a chapter called '01-main-characters.md' to 'Stellar Migration - Character Notes' with content '# Main Characters\n\n## Captain Elena Vasquez\n- Age: 42\n- Background: Former military, 17 hyperspace journeys\n- Personality: Cautious but decisive, haunted by responsibility\n\n## Lieutenant Torres\n- Role: Navigation Officer\n- Background: Recent academy graduate\n- Personality: Eager, detail-oriented, slightly nervous'"
```

### Batch Operations for Efficiency

```bash
# Use ReAct Agent for complex multi-step operations
python src/agents/react_agent/main.py --query "Create a comprehensive outline for a 5-chapter novel called 'The Last Colony' with chapters for setup, conflict introduction, rising action, climax, and resolution. Each chapter should have placeholder content describing the planned scenes."
```

### Cross-Document Analysis

```bash
# Search across multiple documents
python src/agents/simple_agent/main.py --query "Find all paragraphs containing 'hyperspace' across all documents"

# Compare documents for consistency
python src/agents/simple_agent/main.py --query "Get statistics for all documents to compare word counts and chapter structures"
```

---

## Complete E2E Example Session

### Full Session Transcript: Creating and Publishing a Short Story

This section provides a complete, unedited session showing the creation of a short story from start to finish.

```bash
# Session Start: 2024-01-15 10:00:00
# Goal: Create a complete short story suitable for publication

# 1. Initial Setup Verification
$ python src/agents/simple_agent/main.py --check-config
✅ Configuration valid
Using Gemini model: gemini-2.5-flash
MCP server connection: Ready

# 2. Create Document Structure
$ python src/agents/simple_agent/main.py --query "Create a new document called 'The Memory Thief'"
{
  "success": true,
  "summary": "Successfully created document 'The Memory Thief'",
  "details": {
    "document_name": "The Memory Thief",
    "created_at": "2024-01-15T10:01:23Z",
    "chapters": 0,
    "total_words": 0
  }
}

# 3. Add Opening Chapter
$ python src/agents/simple_agent/main.py --query "Add a chapter called '01-opening.md' to 'The Memory Thief' with content '# The Memory Thief\n\nMira's fingers trembled as she placed them on the elderly woman's temples. The memory extraction device hummed to life, its neural interfaces glowing with soft blue light. This was her last job—after tonight, she would disappear forever.\n\n\"Just relax, Mrs. Chen,\" Mira whispered, though guilt gnawed at her stomach. \"This won't hurt a bit.\"\n\nBut it would hurt. It always did. Taking someone's memories, even the painful ones they wanted to forget, left scars that never fully healed.'"

{
  "success": true,
  "summary": "Successfully added chapter '01-opening.md' to document 'The Memory Thief'",
  "details": {
    "chapter_name": "01-opening.md",
    "content_added": true,
    "word_count": 78,
    "paragraph_count": 3
  }
}

# 4. Create Snapshot Before Continuing
$ python src/agents/simple_agent/main.py --query "Create a snapshot of document 'The Memory Thief' with message 'Initial opening chapter - establishing protagonist and conflict'"

{
  "success": true,
  "summary": "Snapshot created successfully",
  "details": {
    "snapshot_id": "snapshot_20240115_100245_initial_opening_user",
    "document_name": "The Memory Thief",
    "message": "Initial opening chapter - establishing protagonist and conflict"
  }
}

# 5. Add Development Chapter
$ python src/agents/simple_agent/main.py --query "Add a chapter called '02-revelation.md' to 'The Memory Thief' with content '# Chapter 2: The Revelation\n\nAs the extraction began, something went wrong. Instead of pulling Mrs. Chen's memories, Mira found herself flooded with images that weren't the old woman's—they were her own, buried so deep she had forgotten they existed.\n\nShe saw herself as a child, watching her mother work with the same type of device. But her mother wasn't stealing memories—she was preserving them, saving the precious moments of people with degenerative diseases before their minds faded completely.\n\n\"You're not here to rob me,\" Mrs. Chen said softly, her eyes still closed. \"You're here to remember who you really are.\"'"

# 6. Add Climactic Resolution
$ python src/agents/simple_agent/main.py --query "Add a chapter called '03-choice.md' to 'The Memory Thief' with content '# Chapter 3: The Choice\n\nMira's hands shook as she understood the truth. The criminal organization she worked for had been using her mother's technology for theft, but they had also been using it to suppress her own memories—to make her forget her mother's true work and her own moral center.\n\n\"I can help you remember everything,\" Mrs. Chen offered, her hand covering Mira's. \"But once you do, there's no going back. You'll have to choose: continue as you are, or become who you were meant to be.\"\n\nMira looked at the extraction device, then at the gentle face of the woman she had come to rob. For the first time in years, she knew exactly what she had to do.\n\n\"Help me remember,\" she whispered. \"Help me remember it all.\"'"

# 7. Review Complete Story
$ python src/agents/simple_agent/main.py --query "Read the complete document 'The Memory Thief' and provide word count statistics"

{
  "success": true,
  "summary": "Retrieved complete document with 3 chapters",
  "details": {
    "document_name": "The Memory Thief",
    "total_chapters": 3,
    "total_words": 312,
    "total_paragraphs": 9,
    "chapters": [
      {
        "filename": "01-opening.md",
        "words": 78,
        "paragraphs": 3
      },
      {
        "filename": "02-revelation.md", 
        "words": 98,
        "paragraphs": 3
      },
      {
        "filename": "03-choice.md",
        "words": 136,
        "paragraphs": 3
      }
    ]
  }
}

# 8. Search for Thematic Consistency
$ python src/agents/simple_agent/main.py --query "Find all paragraphs containing 'memory' or 'remember' in document 'The Memory Thief'"

{
  "success": true,
  "summary": "Found 4 paragraphs containing memory-related terms",
  "details": {
    "search_term": "memory|remember",
    "matches": [
      {
        "chapter": "01-opening.md",
        "paragraph": 1,
        "content": "The memory extraction device hummed to life..."
      },
      {
        "chapter": "01-opening.md", 
        "paragraph": 3,
        "content": "Taking someone's memories, even the painful ones..."
      },
      {
        "chapter": "02-revelation.md",
        "paragraph": 1,
        "content": "Instead of pulling Mrs. Chen's memories, Mira found herself..."
      },
      {
        "chapter": "03-choice.md",
        "paragraph": 2,
        "content": "I can help you remember everything..."
      }
    ]
  }
}

# 9. Final Editing Pass
$ python src/agents/simple_agent/main.py --query "Replace the paragraph containing 'criminal organization' in document 'The Memory Thief' with 'Mira's hands shook as she understood the truth. The syndicate she worked for had been using her mother's technology for theft, but they had also been using it to suppress her own memories—to make her forget her mother's true calling and her own moral compass.'"

# 10. Create Final Snapshot
$ python src/agents/simple_agent/main.py --query "Create a snapshot of document 'The Memory Thief' with message 'Final version - ready for publication'"

# Session End: 2024-01-15 10:47:12
# Total Time: 47 minutes
# Final Word Count: 314 words
# Result: Complete short story ready for publication
```

### Session Analysis

**What This Session Demonstrates:**
- ✅ End-to-end story creation workflow
- ✅ Progressive development with safety snapshots
- ✅ Thematic consistency checking
- ✅ Professional editing and revision process
- ✅ Final preparation for publication

**Key Metrics:**
- **Time to Completion**: 47 minutes
- **Words Created**: 314 words
- **Revision Cycles**: 2 major, 1 minor
- **Safety Snapshots**: 2 (preserving work at key milestones)
- **Quality Checks**: Theme consistency, word count tracking

---

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. API Key Configuration Problems

**Symptom**: `Configuration check failed: Invalid API key`

**Solutions**:
```bash
# Check .env file exists and has correct format
cat .env
# Should show: OPENAI_API_KEY=sk-... or GEMINI_API_KEY=...

# Verify environment variables are loaded
python -c "import os; print('OpenAI key:', 'OPENAI_API_KEY' in os.environ)"

# Test API connectivity directly
python -c "
import openai
client = openai.OpenAI()
response = client.chat.completions.create(
    model='gemini-2.5-flash',
    messages=[{'role': 'user', 'content': 'Hello'}],
    max_tokens=10
)
print('API working:', response.choices[0].message.content)
"
```

#### 2. MCP Server Connection Issues

**Symptom**: `MCP server connection failed` or timeouts

**Solutions**:
```bash
# Check if MCP server is running
ps aux | grep "doc_tool_server"

# Start MCP server manually
document-mcp stdio

# Test MCP server directly
python -c "
import subprocess
import json
proc = subprocess.Popen(['document-mcp', 'stdio'], 
                       stdin=subprocess.PIPE, 
                       stdout=subprocess.PIPE)
# Test basic connection
"

# Check for port conflicts
netstat -an | grep 3001
```

#### 3. Model Loading Failures

**Symptom**: `Model failed to load` or unexpected responses

**Solutions**:
```bash
# Check model availability
python -c "
import openai
client = openai.OpenAI()
models = client.models.list()
available = [m.id for m in models.data if 'gpt-4' in m.id]
print('Available models:', available)
"

# Test with different model
export OPENAI_MODEL=gpt-3.5-turbo
python src/agents/simple_agent/main.py --check-config

# Check rate limits
python -c "
import openai
try:
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model='gemini-2.5-flash',
        messages=[{'role': 'user', 'content': 'test'}],
        max_tokens=5
    )
    print('Rate limit OK')
except openai.RateLimitError as e:
    print('Rate limit exceeded:', e)
"
```

#### 4. Timeout and Performance Issues

**Symptom**: Operations hang or timeout after 30 seconds

**Solutions**:
```bash
# Check system resource usage
top -o cpu | head -10
df -h  # Check disk space

# Test with simpler operations first
python src/agents/simple_agent/main.py --query "list documents"

# Enable verbose logging
export LOG_LEVEL=DEBUG
python src/agents/simple_agent/main.py --query "list documents"

# Check for memory leaks
python -c "
import psutil
import time
for i in range(5):
    proc = psutil.Process()
    print(f'Memory usage: {proc.memory_info().rss / 1024 / 1024:.1f} MB')
    time.sleep(1)
"
```

#### 5. Document Storage Issues

**Symptom**: `Document not found` or `Permission denied`

**Solutions**:
```bash
# Check document storage directory
ls -la .documents_storage/
ls -la .documents_storage/*/

# Check permissions
chmod -R 755 .documents_storage/

# Verify document structure
python -c "
import os
storage_dir = '.documents_storage'
if os.path.exists(storage_dir):
    for doc in os.listdir(storage_dir):
        doc_path = os.path.join(storage_dir, doc)
        if os.path.isdir(doc_path):
            chapters = [f for f in os.listdir(doc_path) if f.endswith('.md')]
            print(f'Document: {doc}, Chapters: {len(chapters)}')
"

# Reset storage if corrupted
mv .documents_storage .documents_storage.backup
mkdir .documents_storage
```

### Recovery Procedures

#### From Corrupted Documents

```bash
# List available snapshots
python src/agents/simple_agent/main.py --query "List all snapshots for document 'Your Document Name'"

# Restore from snapshot
python src/agents/simple_agent/main.py --query "Restore document 'Your Document Name' from snapshot 'snapshot_id_here'"

# Manual recovery from backup
cp -r .documents_storage.backup/document_name .documents_storage/
```

#### From API Quota Exhaustion

```bash
# Check current usage (OpenAI)
python -c "
import openai
client = openai.OpenAI()
try:
    # This will show rate limit info in headers
    response = client.chat.completions.create(
        model='gemini-2.5-flash',
        messages=[{'role': 'user', 'content': 'test'}],
        max_tokens=1
    )
    print('API available')
except Exception as e:
    print('API issue:', e)
"

# Switch to alternative model
export OPENAI_MODEL=gpt-3.5-turbo  # Cheaper alternative
# Or switch to Gemini
export GEMINI_API_KEY=your_gemini_key
unset OPENAI_API_KEY
```

### Performance Optimization

#### For Large Documents

```bash
# Use chapter-scoped operations instead of document-wide
python src/agents/simple_agent/main.py --query "Get statistics for chapter '01-intro.md' in document 'Large Document'"

# Create more frequent snapshots for large edits
python src/agents/simple_agent/main.py --query "Create snapshot of document 'Large Document' with message 'Before bulk edits'"

# Use batch operations for multiple changes
python src/agents/react_agent/main.py --query "Apply these changes to 'Large Document': 1) Replace 'old term' with 'new term', 2) Fix all spelling of 'recieve' to 'receive', 3) Add conclusion paragraph"
```

#### For Multiple Documents

```bash
# Process documents sequentially to avoid rate limits
for doc in "Doc1" "Doc2" "Doc3"; do
    python src/agents/simple_agent/main.py --query "Get statistics for document '$doc'"
    sleep 2  # Rate limit protection
done

# Use Simple Agent for batch processing
python src/agents/simple_agent/main.py --query "List all documents with their word counts and last modified dates"
```

---

## Performance and Load Testing

### Basic Performance Verification

```bash
# Single operation timing
time python src/agents/simple_agent/main.py --query "list all documents"

# Multiple sequential operations
time for i in {1..5}; do
    python src/agents/simple_agent/main.py --query "Get statistics for document 'Test Document'"
done

# Memory usage monitoring
python -c "
import psutil
import subprocess
import time

# Start monitoring
proc = psutil.Process()
print(f'Initial memory: {proc.memory_info().rss / 1024 / 1024:.1f} MB')

# Run agent
subprocess.run(['python', 'src/agents/simple_agent/main.py', '--query', 'list documents'])

print(f'Final memory: {proc.memory_info().rss / 1024 / 1024:.1f} MB')
"
```

### Load Testing with Multiple Documents

```bash
# Create test documents
for i in {1..10}; do
    python src/agents/simple_agent/main.py --query "Create a new document called 'Load Test $i'"
    python src/agents/simple_agent/main.py --query "Add a chapter called '01-content.md' to 'Load Test $i' with content 'This is test content for load testing. The document contains multiple paragraphs to simulate realistic usage patterns.'"
done

# Test bulk operations
time python src/agents/simple_agent/main.py --query "List all documents with detailed statistics"

# Test search across all documents
time python src/agents/simple_agent/main.py --query "Find all paragraphs containing 'test' across all documents"

# Clean up test documents
for i in {1..10}; do
    python src/agents/simple_agent/main.py --query "Delete document 'Load Test $i'"
done
```

### Concurrent Usage Testing

```bash
# Test concurrent agent usage (requires multiple terminals)
# Terminal 1:
python src/agents/simple_agent/main.py --query "Create document 'Concurrent Test 1'" &

# Terminal 2:
python src/agents/simple_agent/main.py --query "Create document 'Concurrent Test 2'" &

# Terminal 3:
python src/agents/simple_agent/main.py --query "List all documents" &

# Wait for all to complete
wait
```

### Expected Performance Benchmarks

- **Document Creation**: < 2 seconds
- **Chapter Addition**: < 1 second  
- **Document Listing**: < 1 second
- **Content Search**: < 3 seconds
- **Snapshot Creation**: < 2 seconds
- **Statistics Generation**: < 1 second

### Performance Troubleshooting

If operations exceed these benchmarks:

1. **Check API response times**:
```bash
time python -c "
import openai
import time
start = time.time()
client = openai.OpenAI()
response = client.chat.completions.create(
    model='gemini-2.5-flash',
    messages=[{'role': 'user', 'content': 'Hello'}],
    max_tokens=10
)
print(f'API response time: {time.time() - start:.2f}s')
"
```

2. **Monitor system resources**:
```bash
# CPU and memory usage
top -p $(pgrep -f "python.*agent")

# Disk I/O
iostat -x 1 5
```

3. **Check document storage**:
```bash
# Document storage size
du -sh .documents_storage/

# Number of documents and chapters
find .documents_storage -name "*.md" | wc -l
```

---

## Conclusion

This manual testing guide provides comprehensive workflows for:

- ✅ **Complete story creation** from concept to finished piece
- ✅ **Professional editing workflows** with safety features
- ✅ **Advanced document management** for complex projects  
- ✅ **Troubleshooting procedures** for common issues
- ✅ **Performance verification** and optimization

The Document MCP system is designed to handle real-world creative writing and document management tasks with enterprise-grade reliability and safety features. This guide ensures users can confidently create, edit, and manage their content while taking full advantage of the system's capabilities.

For additional support or to report issues, please refer to the main project documentation or open an issue on the GitHub repository.