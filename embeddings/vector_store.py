import chromadb
from sentence_transformers import SentenceTransformer
import yaml
import os

class VectorStore:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize Chroma client
        self.chroma_client = chromadb.PersistentClient(
            path=self.config['vector_store']['chroma_path']
        )
        
        # Initialize sentence transformer
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Setup collections
        self.sql_collection = self.chroma_client.get_or_create_collection(
            name=self.config['vector_store']['sql_collection']
        )
        self.schema_collection = self.chroma_client.get_or_create_collection(
            name=self.config['vector_store']['schema_collection']
        )
        
        # Initialize with sample data if collections are empty
        self._initialize_data()

    def _initialize_data(self):
        # Initialize SQL examples if collection is empty
        if self.sql_collection.count() == 0:
            sample_sqls = {
                "sql1": {"sql": "SELECT * FROM users WHERE age > 30", "description": "Get all users over 30"},
                "sql2": {"sql": "SELECT name, email FROM customers", "description": "Get customer names and emails"}
            }
            for id, data in sample_sqls.items():
                embedding = self.model.encode(data["description"]).tolist()
                self.sql_collection.add(
                    ids=[id],
                    embeddings=[embedding],
                    metadatas=[{"sql": data["sql"], "description": data["description"]}],
                    documents=[data["description"]]
                )

        # Initialize schemas if collection is empty
        if self.schema_collection.count() == 0:
            sample_schemas = {
                "users": ["id", "name", "age", "email"],
                "customers": ["id", "name", "email", "phone"]
            }
            for table_name, columns in sample_schemas.items():
                schema_text = f"Table {table_name}: {', '.join(columns)}"
                embedding = self.model.encode(schema_text).tolist()
                self.schema_collection.add(
                    ids=[table_name],
                    embeddings=[embedding],
                    metadatas=[{"columns": columns}],
                    documents=[schema_text]
                )

    def get_similar_sql(self, question, top_k=3):
        query_embedding = self.model.encode(question).tolist()
        results = self.sql_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        similar_sqls = []
        for i in range(len(results['ids'][0])):
            sql = results['metadatas'][0][i]['sql']
            similarity = results['distances'][0][i]
            similar_sqls.append((sql, 1 - similarity))  # Convert distance to similarity
        return similar_sqls

    def get_relevant_schema(self, question):
        query_embedding = self.model.encode(question).tolist()
        results = self.schema_collection.query(
            query_embeddings=[query_embedding],
            n_results=2  # Get top 2 most relevant schemas
        )
        
        schemas = {}
        for i in range(len(results['ids'][0])):
            table_name = results['ids'][0][i]
            columns = results['metadatas'][0][i]['columns']
            schemas[table_name] = columns
        return schemas

    def add_sql_example(self, sql, description):
        embedding = self.model.encode(description).tolist()
        id = f"sql_{self.sql_collection.count() + 1}"
        self.sql_collection.add(
            ids=[id],
            embeddings=[embedding],
            metadatas=[{"sql": sql, "description": description}],
            documents=[description]
        )

    def add_schema(self, table_name, columns):
        schema_text = f"Table {table_name}: {', '.join(columns)}"
        embedding = self.model.encode(schema_text).tolist()
        self.schema_collection.add(
            ids=[table_name],
            embeddings=[embedding],
            metadatas=[{"columns": columns}],
            documents=[schema_text]
        )