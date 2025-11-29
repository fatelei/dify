# InterSystems IRIS Vector Database Integration

This module provides integration with InterSystems IRIS as a vector database for Dify, enabling healthcare organizations to leverage their existing IRIS infrastructure for AI-powered applications.

## Features

- **Native Vector Storage**: Store and retrieve high-dimensional vectors using IRIS's native VECTOR data type
- **Vector Similarity Search**: Efficient similarity search using built-in IRIS vector functions
- **Full-Text Search**: Leverage IRIS's text search capabilities for comprehensive document retrieval
- **Hybrid Search**: Combine vector similarity and full-text search for improved results
- **Healthcare Integration**: Designed specifically for healthcare environments using IRIS
- **Metadata Support**: Flexible JSON metadata storage for document attributes
- **Connection Pooling**: Optimized database connection management
- **Multi-Language Support**: Built-in support for Unicode and multi-language text processing

## Configuration

### Required Environment Variables

```bash
# Connection
IRIS_HOSTNAME=localhost
IRIS_PORT=1972
IRIS_NAMESPACE=USER
IRIS_USERNAME=_SYSTEM
IRIS_PASSWORD=SYS
```

### Optional Configuration

```bash
# Connection settings
IRIS_CONNECTION_TIMEOUT=30
IRIS_QUERY_TIMEOUT=60
IRIS_MAX_CONNECTIONS=10

# Column names (customizable)
IRIS_VECTOR_COLUMN=embedding
IRIS_TEXT_COLUMN=content
IRIS_METADATA_COLUMN=metadata
IRIS_TABLE_PREFIX=vdb_
```

## Usage

### 1. Set IRIS as Vector Store

In your Dify configuration, set:

```bash
VECTOR_STORE=iris
```

### 2. Configure IRIS Connection

Add the following to your environment or `.env` file:

```bash
# Basic connection
IRIS_HOSTNAME=your_iris_server
IRIS_PORT=1972
IRIS_NAMESPACE=YOUR_NAMESPACE
IRIS_USERNAME=your_username
IRIS_PASSWORD=your_password

# Optional performance settings
IRIS_MAX_CONNECTIONS=20
IRIS_CONNECTION_TIMEOUT=60
IRIS_QUERY_TIMEOUT=120
```

### 3. Table Structure

IRIS will automatically create tables with the following structure:

```sql
CREATE TABLE vdb_<collection_name> (
    id UUID PRIMARY KEY DEFAULT UUID(),
    content LONGVARCHAR,
    metadata LONGVARCHAR,
    embedding VECTOR(<dimension>),
    group_id VARCHAR(255),
    doc_id VARCHAR(255),
    document_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for efficient querying
CREATE INDEX vdb_<collection_name>_group_id_idx ON vdb_<collection_name> (group_id);
CREATE INDEX vdb_<collection_name>_doc_id_idx ON vdb_<collection_name> (doc_id);
CREATE INDEX vdb_<collection_name>_document_id_idx ON vdb_<collection_name> (document_id);
```

## Vector Operations

### Vector Similarity Search

IRIS supports various vector distance functions:

```sql
-- Cosine similarity (default, best for normalized embeddings)
SELECT VECTOR_COSINE(embedding, TO_VECTOR(:query_vector)) as similarity

-- Euclidean distance
SELECT VECTOR_EUCLIDEAN(embedding, TO_VECTOR(:query_vector)) as distance

-- Dot product
SELECT VECTOR_DOT_PRODUCT(embedding, TO_VECTOR(:query_vector)) as dot_product
```

### Vector Functions

- `TO_VECTOR(string)`: Convert string representation to VECTOR type
- `VECTOR_COSINE(vector1, vector2)`: Calculate cosine similarity
- `VECTOR_EUCLIDEAN(vector1, vector2)`: Calculate Euclidean distance
- `VECTOR_DOT_PRODUCT(vector1, vector2)`: Calculate dot product

## Full-Text Search Capabilities

IRIS supports comprehensive full-text search with:

### Text Search Patterns

```sql
-- Basic text search
WHERE content LIKE '%search term%'

-- Case-insensitive search
WHERE LOWER(content) LIKE LOWER('%search term%')

-- Multiple conditions
WHERE content LIKE '%machine%' AND content LIKE '%learning%'
```

## Healthcare Integration Benefits

### 1. Existing Infrastructure Leverage
- Use current IRIS deployments without additional database systems
- Maintain data governance and compliance within existing systems
- Reduce operational complexity and costs

### 2. Data Security and Privacy
- Leverage IRIS's robust security features
- Maintain HIPAA compliance within familiar infrastructure
- Control data access through existing IRIS security models

### 3. Performance and Scalability
- Benefit from IRIS's optimized vector operations
- Leverage IRIS's distributed architecture for large-scale deployments
- Utilize IRIS's built-in caching and optimization

### 4. Interoperability
- Seamless integration with existing healthcare applications
- Support for HL7 FHIR and other healthcare data standards
- Compatible with healthcare analytics and reporting tools

## Implementation Details

### Vector Storage Format

Vectors are stored using IRIS's native VECTOR data type:

```sql
-- Store 768-dimensional vector
UPDATE documents SET embedding = TO_VECTOR('[0.1,0.2,0.3,...]') WHERE id = 'doc1';
```

### Metadata Handling

Document metadata is stored as JSON strings:

```sql
-- Store metadata
UPDATE documents SET metadata = '{"doc_id": "doc1", "source": "EHR", "patient_id": "12345"}' WHERE id = 'doc1';
```

### Batch Processing

Documents are processed in configurable batch sizes for optimal performance:

```python
# Default batch size is 64 documents
iris_vector.add_texts(documents, embeddings, batch_size=64)
```

## Performance Optimization

### Connection Management

1. **Connection Pooling**: Configure appropriate pool size based on concurrent users
   ```bash
   IRIS_MAX_CONNECTIONS=20  # Adjust based on usage
   ```

2. **Timeout Settings**: Optimize for your network and query complexity
   ```bash
   IRIS_CONNECTION_TIMEOUT=60   # Connection timeout
   IRIS_QUERY_TIMEOUT=120      # Query timeout
   ```

### Vector Search Optimization

1. **Vector Dimensions**: Use consistent vector dimensions for optimal indexing
2. **Batch Processing**: Process documents in appropriate batch sizes
3. **Index Strategy**: Ensure proper indexes on frequently queried columns

### Query Performance

1. **Filter First**: Apply metadata filters before vector search when possible
2. **Limit Results**: Use appropriate `top_k` values to reduce computational load
3. **Score Threshold**: Set appropriate similarity thresholds to filter results

## Example Healthcare Use Cases

### 1. Clinical Document Search
```python
# Search patient records
medical_docs = iris_vector.search_by_vector(
    query_vector=symptom_embedding,
    document_ids_filter=["patient_records", "clinical_notes"],
    top_k=10
)
```

### 2. Research Paper Similarity
```python
# Find similar research papers
papers = iris_vector.search_by_vector(
    query_vector=research_embedding,
    score_threshold=0.8,
    top_k=5
)
```

### 3. Drug Information Retrieval
```python
# Hybrid search for drug information
drugs = iris_vector.search_by_full_text(
    query="hypertension medication beta blocker",
    document_ids_filter=["drug_database"],
    top_k=15
)
```

## Troubleshooting

### Connection Issues

1. **Verify Configuration**: Ensure all required environment variables are set
2. **Check Network**: Verify connectivity to IRIS server
3. **Authentication**: Confirm namespace, username, and password are correct
4. **Port Accessibility**: Ensure IRIS port (1972 for SuperServer) is accessible

### Vector Operations

1. **Vector Support**: Verify IRIS version supports VECTOR data type
2. **Dimension Consistency**: Ensure all vectors have the same dimensions
3. **Storage Format**: Check vector string format is correct `[x,y,z,...]`

### Performance Issues

1. **Connection Pool**: Monitor connection pool utilization
2. **Query Optimization**: Use appropriate filters and limits
3. **Index Usage**: Verify indexes are being used by queries
4. **Batch Size**: Optimize batch processing for your workload

## Monitoring and Maintenance

### Health Checks

```sql
-- Check table existence
SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'vdb_collection_name';

-- Monitor vector storage
SELECT COUNT(*), AVG(VECTOR_DIMENSIONS(embedding)) FROM vdb_collection_name;
```

### Performance Monitoring

- Monitor connection pool utilization
- Track query response times
- Monitor vector index performance
- Check database storage usage

## Limitations

1. **Vector Dimensions**: All vectors in a collection must have the same dimensions
2. **IRIS Version**: Requires IRIS version with VECTOR data type support
3. **Connection Limits**: Maximum connections determined by IRIS license
4. **Storage Format**: Vectors stored as strings for compatibility
5. **Text Search**: Basic LIKE-based search (no advanced full-text engine)

## Security Considerations

1. **Authentication**: Use secure authentication methods
2. **Encryption**: Enable IRIS encryption for sensitive data
3. **Access Control**: Implement appropriate user permissions
4. **Audit Trail**: Leverage IRIS's audit capabilities
5. **HIPAA Compliance**: Ensure healthcare data protection requirements are met

## References

- [InterSystems IRIS Documentation](https://docs.intersystems.com/)
- [IRIS Vector Functions](https://docs.intersystems.com/irislatest/csp/docbook/DocBook.UI.Page.cls?KEY=RVECTOR_vector)
- [IRIS SQL Reference](https://docs.intersystems.com/irislatest/csp/docbook/DocBook.UI.Page.cls?KEY=RSQL_functions)
- [Healthcare Data Integration](https://docs.intersystems.com/irislatest/csp/docbook/DocBook.UI.Page.cls?KEY=GHL_healthcare)