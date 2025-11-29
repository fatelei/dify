import unittest
from unittest.mock import MagicMock, patch

import pytest

from configs.middleware.vdb.iris_config import IrisConfig
from core.rag.datasource.vdb.iris.iris_vector import IrisVector, IrisVectorFactory
from core.rag.models.document import Document


class TestIrisVector(unittest.TestCase):
    def setUp(self):
        self.config = IrisConfig(
            IRIS_HOSTNAME="localhost",
            IRIS_PORT=1972,
            IRIS_NAMESPACE="USER",
            IRIS_USERNAME="_SYSTEM",
            IRIS_PASSWORD="SYS",
            IRIS_CONNECTION_TIMEOUT=30,
            IRIS_QUERY_TIMEOUT=60,
            IRIS_MAX_CONNECTIONS=10,
        )
        self.collection_name = "test_collection"
        self.group_id = "test_group"

        # Sample documents for testing
        self.sample_documents = [
            Document(
                page_content="This is a test document about AI.",
                metadata={"doc_id": "doc1", "document_id": "dataset1", "source": "test"},
            ),
            Document(
                page_content="Another document about machine learning.",
                metadata={"doc_id": "doc2", "document_id": "dataset1", "source": "test"},
            ),
        ]

        # Sample embeddings
        self.sample_embeddings = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]

    @patch("core.rag.datasource.vdb.iris.iris_vector.create_engine")
    @patch("core.rag.datasource.vdb.iris.iris_vector.sessionmaker")
    def test_init(self, mock_sessionmaker, mock_create_engine):
        """Test IrisVector initialization."""
        # Mock engine and session factory
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_session_factory = MagicMock()
        mock_sessionmaker.return_value = mock_session_factory

        iris_vector = IrisVector(self.collection_name, self.group_id, self.config)

        assert iris_vector.collection_name == self.collection_name
        assert iris_vector._group_id == self.group_id
        assert iris_vector._client_config == self.config
        assert iris_vector._engine == mock_engine
        assert iris_vector._session_local == mock_session_factory
        assert iris_vector.get_type() == "iris"

    @patch("core.rag.datasource.vdb.iris.iris_vector.create_engine")
    @patch("core.rag.datasource.vdb.iris.iris_vector.sessionmaker")
    def test_create_table_name(self, mock_sessionmaker, mock_create_engine):
        """Test table name generation."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_session_factory = MagicMock()
        mock_sessionmaker.return_value = mock_session_factory

        iris_vector = IrisVector(self.collection_name, self.group_id, self.config)
        table_name = iris_vector._create_table_name()

        assert table_name == f"{self.config.IRIS_TABLE_PREFIX}{self.collection_name}"

    @patch("core.rag.datasource.vdb.iris.iris_vector.create_engine")
    @patch("core.rag.datasource.vdb.iris.iris_vector.sessionmaker")
    def test_create_table_sql(self, mock_sessionmaker, mock_create_engine):
        """Test SQL generation for table creation."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_session_factory = MagicMock()
        mock_sessionmaker.return_value = mock_session_factory

        iris_vector = IrisVector(self.collection_name, self.group_id, self.config)
        create_sql = iris_vector._create_table_sql("test_table", 768)

        assert "CREATE TABLE test_table" in create_sql
        assert "VECTOR(768)" in create_sql
        assert iris_vector.config.IRIS_TEXT_COLUMN in create_sql
        assert iris_vector.config.IRIS_METADATA_COLUMN in create_sql
        assert iris_vector.config.IRIS_VECTOR_COLUMN in create_sql

    @patch("core.rag.datasource.vdb.iris.iris_vector.create_engine")
    @patch("core.rag.datasource.vdb.iris.iris_vector.sessionmaker")
    def test_create_index_sql(self, mock_sessionmaker, mock_create_engine):
        """Test SQL generation for index creation."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_session_factory = MagicMock()
        mock_sessionmaker.return_value = mock_session_factory

        iris_vector = IrisVector(self.collection_name, self.group_id, self.config)
        index_sqls = iris_vector._create_index_sql("test_table")

        assert len(index_sqls) == 3  # group_id, doc_id, document_id indexes
        assert all("CREATE INDEX" in sql for sql in index_sqls)
        assert all("test_table" in sql for sql in index_sqls)

    @patch("core.rag.datasource.vdb.iris.iris_vector.create_engine")
    @patch("core.rag.datasource.vdb.iris.iris_vector.sessionmaker")
    @patch("core.rag.datasource.vdb.iris.iris_vector.redis_client")
    def test_create_collection(self, mock_redis, mock_sessionmaker, mock_create_engine):
        """Test collection creation."""
        # Setup Redis mock
        mock_redis.lock.return_value.__enter__ = MagicMock()
        mock_redis.lock.return_value.__exit__ = MagicMock()
        mock_redis.get.return_value = None
        mock_redis.set.return_value = None

        # Mock engine and session
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_session_factory = MagicMock()
        mock_sessionmaker.return_value = mock_session_factory

        mock_session = MagicMock()
        mock_session_factory.return_value = mock_session

        # Mock table doesn't exist
        mock_session.execute.return_value.scalar.return_value = 0

        iris_vector = IrisVector(self.collection_name, self.group_id, self.config)
        iris_vector._create_collection("test_table", 768)

        # Verify Redis lock was used
        mock_redis.lock.assert_called_once()
        # Verify table creation SQL was executed
        assert mock_session.execute.called
        # Verify Redis cache was set
        mock_redis.set.assert_called_once()

    @patch("core.rag.datasource.vdb.iris.iris_vector.create_engine")
    @patch("core.rag.datasource.vdb.iris.iris_vector.sessionmaker")
    @patch("core.rag.datasource.vdb.iris.iris_vector.redis_client")
    def test_create_collection_exists(self, mock_redis, mock_sessionmaker, mock_create_engine):
        """Test collection creation when it already exists."""
        # Setup Redis mock
        mock_redis.lock.return_value.__enter__ = MagicMock()
        mock_redis.lock.return_value.__exit__ = MagicMock()
        mock_redis.get.return_value = 1  # Collection exists in cache

        # Mock engine and session
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_session_factory = MagicMock()
        mock_sessionmaker.return_value = mock_session_factory

        mock_session = MagicMock()
        mock_session_factory.return_value = mock_session

        iris_vector = IrisVector(self.collection_name, self.group_id, self.config)
        iris_vector._create_collection("test_table", 768)

        # Redis lock should be used but no table creation
        mock_redis.lock.assert_called_once()
        mock_redis.get.assert_called_once()
        mock_redis.set.assert_not_called()
        # No table creation should happen
        assert not mock_session.execute.called

    @patch("core.rag.datasource.vdb.iris.iris_vector.create_engine")
    @patch("core.rag.datasource.vdb.iris.iris_vector.sessionmaker")
    def test_text_exists_true(self, mock_sessionmaker, mock_create_engine):
        """Test text exists when document exists."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_session_factory = MagicMock()
        mock_sessionmaker.return_value = mock_session_factory

        mock_session = MagicMock()
        mock_session_factory.return_value = mock_session

        # Mock document exists
        mock_session.execute.return_value.scalar.return_value = 1

        iris_vector = IrisVector(self.collection_name, self.group_id, self.config)
        exists = iris_vector.text_exists("doc1")

        assert exists

    @patch("core.rag.datasource.vdb.iris.iris_vector.create_engine")
    @patch("core.rag.datasource.vdb.iris.iris_vector.sessionmaker")
    def test_text_exists_false(self, mock_sessionmaker, mock_create_engine):
        """Test text exists when document doesn't exist."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_session_factory = MagicMock()
        mock_sessionmaker.return_value = mock_session_factory

        mock_session = MagicMock()
        mock_session_factory.return_value = mock_session

        # Mock document doesn't exist
        mock_session.execute.return_value.scalar.return_value = 0

        iris_vector = IrisVector(self.collection_name, self.group_id, self.config)
        exists = iris_vector.text_exists("doc1")

        assert not exists

    @patch("core.rag.datasource.vdb.iris.iris_vector.create_engine")
    @patch("core.rag.datasource.vdb.iris.iris_vector.sessionmaker")
    @patch("core.rag.datasource.vdb.iris.iris_vector.redis_client")
    def test_create(self, mock_redis, mock_sessionmaker, mock_create_engine):
        """Test vector creation with documents and embeddings."""
        # Setup Redis mock
        mock_redis.lock.return_value.__enter__ = MagicMock()
        mock_redis.lock.return_value.__exit__ = MagicMock()
        mock_redis.get.return_value = None
        mock_redis.set.return_value = None

        # Mock engine and session
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_session_factory = MagicMock()
        mock_sessionmaker.return_value = mock_session_factory

        mock_session = MagicMock()
        mock_session_factory.return_value = mock_session

        # Mock table doesn't exist
        mock_session.execute.return_value.scalar.return_value = 0

        iris_vector = IrisVector(self.collection_name, self.group_id, self.config)
        iris_vector.create(self.sample_documents, self.sample_embeddings)

        # Should create collection and add documents
        assert mock_session.execute.called
        assert mock_session.commit.called

    @patch("core.rag.datasource.vdb.iris.iris_vector.create_engine")
    @patch("core.rag.datasource.vdb.iris.iris_vector.sessionmaker")
    def test_add_texts(self, mock_sessionmaker, mock_create_engine):
        """Test adding texts with embeddings."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_session_factory = MagicMock()
        mock_sessionmaker.return_value = mock_session_factory

        mock_session = MagicMock()
        mock_session_factory.return_value = mock_session

        iris_vector = IrisVector(self.collection_name, self.group_id, self.config)
        result = iris_vector.add_texts(self.sample_documents, self.sample_embeddings)

        assert len(result) == 2
        mock_session.execute.assert_called()
        mock_session.commit.assert_called()

    @patch("core.rag.datasource.vdb.iris.iris_vector.create_engine")
    @patch("core.rag.datasource.vdb.iris.iris_vector.sessionmaker")
    def test_delete_by_ids(self, mock_sessionmaker, mock_create_engine):
        """Test deleting documents by IDs."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_session_factory = MagicMock()
        mock_sessionmaker.return_value = mock_session_factory

        mock_session = MagicMock()
        mock_session_factory.return_value = mock_session

        iris_vector = IrisVector(self.collection_name, self.group_id, self.config)
        iris_vector.delete_by_ids(["doc1", "doc2"])

        # Verify delete SQL was executed
        mock_session.execute.assert_called()
        mock_session.commit.assert_called()

        # Check the delete statement
        execute_calls = mock_session.execute.call_args_list
        delete_calls = [call for call in execute_calls if "DELETE" in str(call)]
        assert len(delete_calls) == 1

    @patch("core.rag.datasource.vdb.iris.iris_vector.create_engine")
    @patch("core.rag.datasource.vdb.iris.iris_vector.sessionmaker")
    def test_delete_by_metadata_field(self, mock_sessionmaker, mock_create_engine):
        """Test deleting documents by metadata field."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_session_factory = MagicMock()
        mock_sessionmaker.return_value = mock_session_factory

        mock_session = MagicMock()
        mock_session_factory.return_value = mock_session

        iris_vector = IrisVector(self.collection_name, self.group_id, self.config)
        iris_vector.delete_by_metadata_field("document_id", "dataset1")

        # Verify delete SQL was executed
        mock_session.execute.assert_called()
        mock_session.commit.assert_called()

        # Check the delete statement contains metadata filtering
        execute_calls = mock_session.execute.call_args_list
        delete_calls = [call for call in execute_calls if "DELETE" in str(call)]
        assert len(delete_calls) == 1

    @patch("core.rag.datasource.vdb.iris.iris_vector.create_engine")
    @patch("core.rag.datasource.vdb.iris.iris_vector.sessionmaker")
    def test_delete(self, mock_sessionmaker, mock_create_engine):
        """Test deleting entire collection."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_session_factory = MagicMock()
        mock_sessionmaker.return_value = mock_session_factory

        mock_session = MagicMock()
        mock_session_factory.return_value = mock_session

        iris_vector = IrisVector(self.collection_name, self.group_id, self.config)
        iris_vector.delete()

        # Verify delete SQL was executed for group
        mock_session.execute.assert_called()
        mock_session.commit.assert_called()

    @patch("core.rag.datasource.vdb.iris.iris_vector.create_engine")
    @patch("core.rag.datasource.vdb.iris.iris_vector.sessionmaker")
    def test_search_by_vector(self, mock_sessionmaker, mock_create_engine):
        """Test vector similarity search."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_session_factory = MagicMock()
        mock_sessionmaker.return_value = mock_session_factory

        mock_session = MagicMock()
        mock_session_factory.return_value = mock_session

        # Mock search results
        mock_result = MagicMock()
        mock_result.similarity = 0.9
        mock_session.execute.return_value.fetchall.return_value = [mock_result]

        # Mock attribute access
        def mock_getattr(obj, attr, default=None):
            if attr == self.config.IRIS_TEXT_COLUMN:
                return "Test document"
            elif attr == self.config.IRIS_METADATA_COLUMN:
                return '{"doc_id": "doc1", "source": "test"}'
            return getattr(obj, attr, default)

        with patch.object(MagicMock, "__getattr__", side_effect=mock_getattr):
            iris_vector = IrisVector(self.collection_name, self.group_id, self.config)
            query_vector = [0.1, 0.2, 0.3, 0.4]
            docs = iris_vector.search_by_vector(query_vector, top_k=5)

            assert len(docs) == 1
            assert docs[0].page_content == "Test document"
            assert docs[0].metadata["score"] == 0.9

    @patch("core.rag.datasource.vdb.iris.iris_vector.create_engine")
    @patch("core.rag.datasource.vdb.iris.iris_vector.sessionmaker")
    def test_search_by_full_text(self, mock_sessionmaker, mock_create_engine):
        """Test full-text search."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_session_factory = MagicMock()
        mock_sessionmaker.return_value = mock_session_factory

        mock_session = MagicMock()
        mock_session_factory.return_value = mock_session

        # Mock search results
        mock_result = MagicMock()
        mock_session.execute.return_value.fetchall.return_value = [mock_result]

        # Mock attribute access
        def mock_getattr(obj, attr, default=None):
            if attr == self.config.IRIS_TEXT_COLUMN:
                return "This document contains machine learning content"
            elif attr == self.config.IRIS_METADATA_COLUMN:
                return '{"doc_id": "doc1", "source": "test"}'
            return getattr(obj, attr, default)

        with patch.object(MagicMock, "__getattr__", side_effect=mock_getattr):
            iris_vector = IrisVector(self.collection_name, self.group_id, self.config)
            docs = iris_vector.search_by_full_text("machine learning", top_k=5)

            assert len(docs) == 1
            assert docs[0].page_content == "This document contains machine learning content"

    @patch("core.rag.datasource.vdb.iris.iris_vector.create_engine")
    @patch("core.rag.datasource.vdb.iris.iris_vector.sessionmaker")
    def test_vector_to_string(self, mock_sessionmaker, mock_create_engine):
        """Test vector to string conversion."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_session_factory = MagicMock()
        mock_sessionmaker.return_value = mock_session_factory

        iris_vector = IrisVector(self.collection_name, self.group_id, self.config)
        vector_string = iris_vector._vector_to_string([0.1, 0.2, 0.3])

        assert vector_string == "[0.1,0.2,0.3]"

    @patch("core.rag.datasource.vdb.iris.iris_vector.create_engine")
    def test_to_index_struct(self, mock_create_engine):
        """Test index structure generation."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        iris_vector = IrisVector(self.collection_name, self.group_id, self.config)
        index_struct = iris_vector.to_index_struct()

        assert index_struct["type"] == "iris"
        assert index_struct["vector_store"]["class_prefix"] == self.collection_name


class TestIrisVectorFactory(unittest.TestCase):
    @patch("core.rag.datasource.vdb.iris.iris_vector.IrisVector")
    @patch("core.rag.datasource.vdb.iris.iris_vector.IrisConfig")
    @patch("core.rag.datasource.vdb.iris.iris_vector.Dataset.gen_collection_name_by_id")
    def test_init_vector_without_collection_binding(self, mock_gen_name, mock_config, mock_vector):
        """Test factory initialization without collection binding."""
        # Setup mocks
        mock_gen_name.return_value = "test_collection_name"
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance
        mock_vector_instance = MagicMock()
        mock_vector.return_value = mock_vector_instance

        # Create dataset without collection binding
        dataset = MagicMock()
        dataset.collection_binding_id = None
        dataset.index_struct_dict = None
        dataset.id = "test_dataset_id"
        dataset.index_struct = None

        # Mock attributes
        attributes = ["doc_id", "dataset_id"]

        # Mock embeddings
        embeddings = MagicMock()

        factory = IrisVectorFactory()
        result = factory.init_vector(dataset, attributes, embeddings)

        # Verify method calls
        mock_gen_name.assert_called_once_with("test_dataset_id")
        mock_config.assert_called_once()
        mock_vector.assert_called_once()

        assert result == mock_vector_instance

    def test_gen_index_struct_dict(self):
        """Test index structure dictionary generation."""
        factory = IrisVectorFactory()
        result = factory.gen_index_struct_dict("iris", "test_collection")

        expected = {"type": "iris", "vector_store": {"class_prefix": "test_collection"}}
        assert result == expected


@pytest.mark.parametrize(
    "invalid_config_override",
    [
        {"IRIS_HOSTNAME": ""},  # Test empty host
        {"IRIS_PORT": 0},  # Test invalid port
        {"IRIS_MAX_CONNECTIONS": 0},  # Test invalid max connections
        {"IRIS_CONNECTION_TIMEOUT": 0},  # Test invalid timeout
        {"IRIS_QUERY_TIMEOUT": 0},  # Test invalid timeout
    ],
)
def test_config_validation_parametrized(invalid_config_override):
    """Test configuration validation for various invalid inputs using parametrize."""
    config = {
        "IRIS_HOSTNAME": "localhost",
        "IRIS_PORT": 1972,
        "IRIS_NAMESPACE": "USER",
        "IRIS_USERNAME": "_SYSTEM",
        "IRIS_PASSWORD": "SYS",
        "IRIS_CONNECTION_TIMEOUT": 30,
        "IRIS_QUERY_TIMEOUT": 60,
        "IRIS_MAX_CONNECTIONS": 10,
        "IRIS_VECTOR_COLUMN": "embedding",
        "IRIS_TEXT_COLUMN": "content",
        "IRIS_METADATA_COLUMN": "metadata",
        "IRIS_TABLE_PREFIX": "vdb_",
    }
    config.update(invalid_config_override)

    with pytest.raises(ValueError):
        IrisConfig(**config)


if __name__ == "__main__":
    unittest.main()
