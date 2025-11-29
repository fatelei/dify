from pydantic import Field, PositiveInt
from pydantic_settings import BaseSettings


class IrisConfig(BaseSettings):
    """
    Configuration settings for InterSystems IRIS vector database
    """

    IRIS_HOSTNAME: str = Field(
        description="Hostname or IP address of the IRIS server",
        default="localhost",
    )

    IRIS_PORT: PositiveInt = Field(
        description=(
            "Port number for IRIS server connection (default is 1972 for SuperServer, 52773 for web connection)"
        ),
        default=1972,
    )

    IRIS_NAMESPACE: str = Field(
        description="IRIS namespace to connect to",
        default="USER",
    )

    IRIS_USERNAME: str = Field(
        description="Username for IRIS authentication",
        default="_SYSTEM",
    )

    IRIS_PASSWORD: str = Field(
        description="Password for IRIS authentication",
        default="SYS",
    )

    IRIS_CONNECTION_TIMEOUT: PositiveInt = Field(
        description="Timeout in seconds for IRIS connection operations",
        default=30,
    )

    IRIS_QUERY_TIMEOUT: PositiveInt = Field(
        description="Timeout in seconds for IRIS query operations",
        default=60,
    )

    IRIS_MAX_CONNECTIONS: PositiveInt = Field(
        description="Maximum number of connections in the connection pool",
        default=10,
    )

    IRIS_VECTOR_COLUMN: str = Field(
        description="Name of the column to store vector embeddings",
        default="embedding",
    )

    IRIS_TEXT_COLUMN: str = Field(
        description="Name of the column to store document text content",
        default="content",
    )

    IRIS_METADATA_COLUMN: str = Field(
        description="Name of the column to store document metadata",
        default="metadata",
    )

    IRIS_TABLE_PREFIX: str = Field(
        description="Prefix for table names used to store vectors",
        default="vdb_",
    )
