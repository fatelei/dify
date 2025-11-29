import json
import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from configs.middleware.vdb.iris_config import IrisConfig
from core.rag.datasource.vdb.vector_base import BaseVector
from core.rag.datasource.vdb.vector_factory import AbstractVectorFactory
from core.rag.datasource.vdb.vector_type import VectorType
from core.rag.embedding.embedding_base import Embeddings
from core.rag.models.document import Document
from extensions.ext_redis import redis_client
from models.dataset import Dataset, DatasetCollectionBinding

if TYPE_CHECKING:
    pass


class IrisVector(BaseVector):
    def __init__(self, collection_name: str, group_id: str, config: IrisConfig):
        super().__init__(collection_name)
        self._client_config = config
        self._group_id = group_id
        self._engine = None
        self._session_local = None
        self._init_connection()

    def _init_connection(self):
        """Initialize IRIS database connection"""
        try:
            # Construct connection string for IRIS
            connection_string = (
                f"iris://{self._client_config.IRIS_USERNAME}:{self._client_config.IRIS_PASSWORD}@"
                f"{self._client_config.IRIS_HOSTNAME}:{self._client_config.IRIS_PORT}/{self._client_config.IRIS_NAMESPACE}"
            )

            # Create SQLAlchemy engine
            self._engine = create_engine(
                connection_string,
                pool_size=self._client_config.IRIS_MAX_CONNECTIONS,
                pool_timeout=self._client_config.IRIS_CONNECTION_TIMEOUT,
                query_timeout=self._client_config.IRIS_QUERY_TIMEOUT,
            )

            # Create session factory
            self._session_local = sessionmaker(bind=self._engine)

        except Exception as e:
            raise ConnectionError(f"Failed to connect to IRIS database: {str(e)}")

    def _get_session(self):
        """Get a database session"""
        return self._session_local()

    def _create_table_name(self) -> str:
        """Generate table name for the collection"""
        return f"{self._client_config.IRIS_TABLE_PREFIX}{self._collection_name}"

    def _create_table_sql(self, table_name: str, vector_dimension: int) -> str:
        """Generate SQL for creating vector table"""
        return f"""
        CREATE TABLE {table_name} (
            id UUID PRIMARY KEY DEFAULT UUID(),
            {self._client_config.IRIS_TEXT_COLUMN} LONGVARCHAR,
            {self._client_config.IRIS_METADATA_COLUMN} LONGVARCHAR,
            {self._client_config.IRIS_VECTOR_COLUMN} VECTOR({vector_dimension}),
            group_id VARCHAR(255),
            doc_id VARCHAR(255),
            document_id VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """

    def _create_index_sql(self, table_name: str) -> list[str]:
        """Generate SQL statements for creating indexes"""
        indexes = [
            f"CREATE INDEX {table_name}_group_id_idx ON {table_name} (group_id)",
            f"CREATE INDEX {table_name}_doc_id_idx ON {table_name} (doc_id)",
            f"CREATE INDEX {table_name}_document_id_idx ON {table_name} (document_id)",
        ]
        return indexes

    def get_type(self) -> str:
        return VectorType.IRIS

    def to_index_struct(self):
        return {"type": self.get_type(), "vector_store": {"class_prefix": self._collection_name}}

    def create(self, texts: list[Document], embeddings: list[list[float]], **kwargs):
        if not texts or not embeddings:
            return

        # Get embedding vector size
        vector_size = len(embeddings[0])
        table_name = self._create_table_name()

        # Create table and indexes
        self._create_collection(table_name, vector_size)

        # Add texts with embeddings
        self.add_texts(texts, embeddings, **kwargs)

    def _create_collection(self, table_name: str, vector_size: int):
        """Create IRIS table for vector storage"""
        lock_name = f"vector_indexing_lock_{table_name}"

        with redis_client.lock(lock_name, timeout=20):
            collection_exist_cache_key = f"vector_indexing_{self._collection_name}"
            if redis_client.get(collection_exist_cache_key):
                return

            session = self._get_session()
            try:
                # Check if table exists
                result = session.execute(
                    text(f"SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = '{table_name}'")
                )
                table_exists = result.scalar() > 0

                if not table_exists:
                    # Create table
                    create_table_sql = self._create_table_sql(table_name, vector_size)
                    session.execute(text(create_table_sql))

                    # Create indexes
                    for index_sql in self._create_index_sql(table_name):
                        try:
                            session.execute(text(index_sql))
                        except Exception:
                            # Some indexes might fail, continue anyway
                            pass

                    session.commit()

                redis_client.set(collection_exist_cache_key, 1, ex=3600)

            except Exception as e:
                session.rollback()
                raise e
            finally:
                session.close()

    def add_texts(self, documents: list[Document], embeddings: list[list[float]], **kwargs):
        if not documents or not embeddings:
            return []

        table_name = self._create_table_name()
        session = self._get_session()

        try:
            added_ids = []

            for doc, embedding in zip(documents, embeddings):
                doc_id = str(uuid.uuid4())

                # Prepare metadata JSON
                metadata = doc.metadata or {}
                metadata["doc_id"] = doc_id
                metadata_json = json.dumps(metadata)

                # Insert document with embedding
                insert_sql = f"""
                INSERT INTO {table_name}
                (id, {self._client_config.IRIS_TEXT_COLUMN}, {self._client_config.IRIS_METADATA_COLUMN},
                 {self._client_config.IRIS_VECTOR_COLUMN}, group_id, doc_id, document_id)
                VALUES (:id, :content, :metadata, TO_VECTOR(:embedding), :group_id, :doc_id, :document_id)
                """

                session.execute(
                    text(insert_sql),
                    {
                        "id": doc_id,
                        "content": doc.page_content,
                        "metadata": metadata_json,
                        "embedding": str(embedding),
                        "group_id": self._group_id,
                        "doc_id": doc_id,
                        "document_id": metadata.get("document_id", ""),
                    },
                )

                added_ids.append(doc_id)

            session.commit()
            return added_ids

        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def text_exists(self, id: str) -> bool:
        table_name = self._create_table_name()
        session = self._get_session()

        try:
            result = session.execute(text(f"SELECT COUNT(*) FROM {table_name} WHERE id = :id"), {"id": id})
            return result.scalar() > 0
        except Exception:
            return False
        finally:
            session.close()

    def delete_by_ids(self, ids: list[str]):
        if not ids:
            return

        table_name = self._create_table_name()
        session = self._get_session()

        try:
            placeholders = ",".join([f":id_{i}" for i in range(len(ids))])
            params = {f"id_{i}": doc_id for i, doc_id in enumerate(ids)}

            delete_sql = f"DELETE FROM {table_name} WHERE id IN ({placeholders})"
            session.execute(text(delete_sql), params)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def delete_by_metadata_field(self, key: str, value: str):
        table_name = self._create_table_name()
        session = self._get_session()

        try:
            delete_sql = f"""
            DELETE FROM {table_name}
            WHERE {self._client_config.IRIS_METADATA_COLUMN} LIKE :pattern
            AND group_id = :group_id
            """
            session.execute(text(delete_sql), {"pattern": f'%"{key}":"{value}"%', "group_id": self._group_id})
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def delete(self):
        table_name = self._create_table_name()
        session = self._get_session()

        try:
            delete_sql = f"DELETE FROM {table_name} WHERE group_id = :group_id"
            session.execute(text(delete_sql), {"group_id": self._group_id})
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def _vector_to_string(self, vector: list[float]) -> str:
        """Convert vector list to string representation"""
        return "[" + ",".join([str(x) for x in vector]) + "]"

    def search_by_vector(self, query_vector: list[float], **kwargs: Any) -> list[Document]:
        table_name = self._create_table_name()
        session = self._get_session()

        try:
            score_threshold = float(kwargs.get("score_threshold") or 0.0)
            top_k = kwargs.get("top_k", 4)

            # Check for document filter
            document_ids_filter = kwargs.get("document_ids_filter")
            where_clause = "WHERE group_id = :group_id"
            params = {"group_id": self._group_id, "query_vector": self._vector_to_string(query_vector)}

            if document_ids_filter:
                placeholders = ",".join([f":doc_id_{i}" for i in range(len(document_ids_filter))])
                for i, doc_id in enumerate(document_ids_filter):
                    params[f"doc_id_{i}"] = doc_id
                where_clause += f" AND document_id IN ({placeholders})"

            # Perform vector similarity search using IRIS VECTOR functions
            search_sql = f"""
            SELECT *,
                   VECTOR_COSINE({self._client_config.IRIS_VECTOR_COLUMN}, TO_VECTOR(:query_vector)) as similarity
            FROM {table_name}
            {where_clause}
            ORDER BY similarity DESC
            LIMIT {top_k}
            """

            result = session.execute(text(search_sql), params)
            rows = result.fetchall()

            docs = []
            for row in rows:
                similarity = float(row.similarity) if hasattr(row, "similarity") else 1.0

                if similarity >= score_threshold:
                    # Parse metadata
                    try:
                        metadata = json.loads(getattr(row, self._client_config.IRIS_METADATA_COLUMN, "{}"))
                    except (json.JSONDecodeError, AttributeError):
                        metadata = {}

                    metadata["score"] = similarity

                    doc = Document(
                        page_content=getattr(row, self._client_config.IRIS_TEXT_COLUMN, ""),
                        metadata=metadata,
                    )
                    docs.append(doc)

            # Sort by similarity score in descending order
            docs = sorted(docs, key=lambda x: x.metadata.get("score", 0), reverse=True)
            return docs

        except Exception as e:
            # If table doesn't exist or other error, return empty list
            return []
        finally:
            session.close()

    def search_by_full_text(self, query: str, **kwargs: Any) -> list[Document]:
        table_name = self._create_table_name()
        session = self._get_session()

        try:
            top_k = kwargs.get("top_k", 2)

            # Check for document filter
            document_ids_filter = kwargs.get("document_ids_filter")
            where_clause = f"WHERE {self._client_config.IRIS_TEXT_COLUMN} LIKE :query AND group_id = :group_id"
            params = {"query": f"%{query}%", "group_id": self._group_id}

            if document_ids_filter:
                placeholders = ",".join([f":doc_id_{i}" for i in range(len(document_ids_filter))])
                for i, doc_id in enumerate(document_ids_filter):
                    params[f"doc_id_{i}"] = doc_id
                where_clause += f" AND document_id IN ({placeholders})"

            # Perform full-text search
            search_sql = f"""
            SELECT *
            FROM {table_name}
            {where_clause}
            LIMIT {top_k}
            """

            result = session.execute(text(search_sql), params)
            rows = result.fetchall()

            docs = []
            for row in rows:
                # Parse metadata
                try:
                    metadata = json.loads(getattr(row, self._client_config.IRIS_METADATA_COLUMN, "{}"))
                except (json.JSONDecodeError, AttributeError):
                    metadata = {}

                doc = Document(
                    page_content=getattr(row, self._client_config.IRIS_TEXT_COLUMN, ""),
                    metadata=metadata,
                )
                docs.append(doc)

            return docs

        except Exception as e:
            # If table doesn't exist or other error, return empty list
            return []
        finally:
            session.close()


class IrisVectorFactory(AbstractVectorFactory):
    def init_vector(self, dataset: Dataset, attributes: list, embeddings: Embeddings) -> IrisVector:
        from configs import dify_config

        if dataset.collection_binding_id:
            from sqlalchemy import select

            from extensions.ext_database import db

            stmt = select(DatasetCollectionBinding).where(DatasetCollectionBinding.id == dataset.collection_binding_id)
            dataset_collection_binding = db.session.scalars(stmt).one_or_none()
            if dataset_collection_binding:
                collection_name = dataset_collection_binding.collection_name
            else:
                raise ValueError("Dataset Collection Bindings does not exist!")
        else:
            if dataset.index_struct_dict:
                class_prefix: str = dataset.index_struct_dict["vector_store"]["class_prefix"]
                collection_name = class_prefix
            else:
                dataset_id = dataset.id
                collection_name = Dataset.gen_collection_name_by_id(dataset_id)

        if not dataset.index_struct_dict:
            dataset.index_struct = json.dumps(self.gen_index_struct_dict(VectorType.IRIS, collection_name))

        return IrisVector(
            collection_name=collection_name,
            group_id=dataset.id,
            config=IrisConfig(
                IRIS_HOSTNAME=getattr(dify_config, "IRIS_HOSTNAME", "localhost"),
                IRIS_PORT=getattr(dify_config, "IRIS_PORT", 1972),
                IRIS_NAMESPACE=getattr(dify_config, "IRIS_NAMESPACE", "USER"),
                IRIS_USERNAME=getattr(dify_config, "IRIS_USERNAME", "_SYSTEM"),
                IRIS_PASSWORD=getattr(dify_config, "IRIS_PASSWORD", "SYS"),
                IRIS_CONNECTION_TIMEOUT=getattr(dify_config, "IRIS_CONNECTION_TIMEOUT", 30),
                IRIS_QUERY_TIMEOUT=getattr(dify_config, "IRIS_QUERY_TIMEOUT", 60),
                IRIS_MAX_CONNECTIONS=getattr(dify_config, "IRIS_MAX_CONNECTIONS", 10),
                IRIS_VECTOR_COLUMN=getattr(dify_config, "IRIS_VECTOR_COLUMN", "embedding"),
                IRIS_TEXT_COLUMN=getattr(dify_config, "IRIS_TEXT_COLUMN", "content"),
                IRIS_METADATA_COLUMN=getattr(dify_config, "IRIS_METADATA_COLUMN", "metadata"),
                IRIS_TABLE_PREFIX=getattr(dify_config, "IRIS_TABLE_PREFIX", "vdb_"),
            ),
        )
