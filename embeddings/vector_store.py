import redis
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.index import IndexDefinition
from langchain.embeddings import SentenceTransformerEmbeddings
import yaml
import numpy as np
import json
import os
from datetime import datetime

class VectorStore:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize Redis client
        self.redis_client = redis.Redis(
            host=self.config['vector_store']['redis_host'],
            port=self.config['vector_store']['redis_port'],
            db=0,
            decode_responses=True
        )
        
        # Initialize SentenceTransformerEmbeddings
        self.embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
        self.vector_dim = 384  # Dimension of all-MiniLM-L6-v2 embeddings
        
        # Setup Redis indices
        self._create_indices()
        
        # Initialize with sample data if empty
        self._initialize_data()

    def _create_indices(self):
        # Create SQL examples index
        try:
            self.redis_client.ft('sql_idx').info()
        except:
            self.redis_client.ft('sql_idx').create_index(
                [
                    TextField('description'),
                    TextField('sql'),
                    VectorField('embedding', 'HNSW', {
                        'TYPE': 'FLOAT32',
                        'DIM': self.vector_dim,
                        'DISTANCE_METRIC': 'COSINE'
                    })
                ],
                definition=IndexDefinition(prefix=['sql:'])
            )

        # Create schema index
        try:
            self.redis_client.ft('schema_idx').info()
        except:
            self.redis_client.ft('schema_idx').create_index(
                [
                    TextField('schema_text'),
                    VectorField('embedding', 'HNSW', {
                        'TYPE': 'FLOAT32',
                        'DIM': self.vector_dim,
                        'DISTANCE_METRIC': 'COSINE'
                    })
                ],
                definition=IndexDefinition(prefix=['schema:'])
            )

    def _initialize_data(self):
        # Initialize SQL examples if empty
        if self.redis_client.dbsize() == 0:
            sample_sqls = {
                "sql1": {"sql": "SELECT * FROM users WHERE age > 30", "description": "Get all users over 30"},
                "sql2": {"sql": "SELECT name, email FROM customers", "description": "Get customer names and emails"}
            }
            for id, data in sample_sqls.items():
                embedding = np.array(self.embeddings.embed_query(data["description"])).astype(np.float32).tobytes()
                self.redis_client.hset(f'sql:{id}', mapping={
                    'description': data["description"],
                    'sql': data["sql"],
                    'embedding': embedding
                })

        # Initialize schemas if empty
        if self.redis_client.keys('schema:*') == []:
            sample_schemas = {
                "users": ["id", "name", "age", "email"],
                "customers": ["id", "name", "email", "phone"]
            }
            for table_name, columns in sample_schemas.items():
                schema_text = f"Table {table_name}: {', '.join(columns)}"
                embedding = np.array(self.embeddings.embed_query(schema_text)).astype(np.float32).tobytes()
                self.redis_client.hset(f'schema:{table_name}', mapping={
                    'schema_text': schema_text,
                    'columns': json.dumps(columns),
                    'embedding': embedding
                })

    def get_similar_sql(self, question, top_k=3):
        query_embedding = np.array(self.embeddings.embed_query(question)).astype(np.float32).tobytes()
        query = f'*=>[KNN {top_k} @embedding $vec AS similarity]'
        
        results = self.redis_client.ft('sql_idx').search(
            query,
            query_params={'vec': query_embedding}
        ).docs
        
        similar_sqls = []
        for doc in results:
            sql = doc.sql
            similarity = float(doc.similarity)
            similar_sqls.append((sql, similarity))
        return similar_sqls

    def get_relevant_schema(self, question):
        query_embedding = np.array(self.embeddings.embed_query(question)).astype(np.float32).tobytes()
        query = f'*=>[KNN 2 @embedding $vec AS similarity]'
        
        results = self.redis_client.ft('schema_idx').search(
            query,
            query_params={'vec': query_embedding}
        ).docs
        
        schemas = {}
        for doc in results:
            table_name = doc.id.split(':')[1]
            columns = json.loads(doc.columns)
            schemas[table_name] = columns
        return schemas

    def add_sql_example(self, sql, description):
        embedding = np.array(self.embeddings.embed_query(description)).astype(np.float32).tobytes()
        id = f"sql_{self.redis_client.dbsize() + 1}"
        self.redis_client.hset(f'sql:{id}', mapping={
            'description': description,
            'sql': sql,
            'embedding': embedding
        })

    def add_schema(self, table_name, columns):
        schema_text = f"Table {table_name}: {', '.join(columns)}"
        embedding = np.array(self.embeddings.embed_query(schema_text)).astype(np.float32).tobytes()
        self.redis_client.hset(f'schema:{table_name}', mapping={
            'schema_text': schema_text,
            'columns': json.dumps(columns),
            'embedding': embedding
        })

def create_sample_config():
    """Create a sample configuration file"""
    config = {
        'vector_store': {
            'redis_host': 'localhost',
            'redis_port': 6379
        }
    }
    
    with open('config.yaml', 'w') as f:
        yaml.safe_dump(config, f)
    return 'config.yaml'

def main():
    # Create sample config if it doesn't exist
    config_path = 'config.yaml'
    if not os.path.exists(config_path):
        config_path = create_sample_config()
    
    # Initialize VectorStore
    print("Initializing VectorStore...")
    vector_store = VectorStore(config_path)
    
    # Test 1: Check initial data loading
    print("\nTest 1: Checking initial data")
    print(f"Number of SQL examples: {len(vector_store.redis_client.keys('sql:*'))}")
    print(f"Number of schemas: {len(vector_store.redis_client.keys('schema:*'))}")
    
    # Test 2: Add new SQL example
    print("\nTest 2: Adding new SQL example")
    new_sql = "SELECT id, name FROM employees WHERE salary > 50000"
    new_desc = "Get employees with salary over 50k"
    vector_store.add_sql_example(new_sql, new_desc)
    print(f"Added SQL: {new_sql}")
    print(f"New total SQL examples: {len(vector_store.redis_client.keys('sql:*'))}")
    
    # Test 3: Add new schema
    print("\nTest 3: Adding new schema")
    new_table = "employees"
    new_columns = ["id", "name", "salary", "department"]
    vector_store.add_schema(new_table, new_columns)
    print(f"Added schema for table '{new_table}' with columns: {new_columns}")
    print(f"New total schemas: {len(vector_store.redis_client.keys('schema:*'))}")
    
    # Test 4: Get similar SQL queries
    print("\nTest 4: Finding similar SQL queries")
    test_question = "Show me employees earning more than 40k"
    similar_sqls = vector_store.get_similar_sql(test_question)
    print(f"Question: {test_question}")
    print("Similar SQL queries found:")
    for sql, similarity in similar_sqls:
        print(f"SQL: {sql} (Similarity: {similarity:.3f})")
    
    # Test 5: Get relevant schemas
    print("\nTest 5: Finding relevant schemas")
    schema_question = "What tables have salary information?"
    relevant_schemas = vector_store.get_relevant_schema(schema_question)
    print(f"Question: {schema_question}")
    print("Relevant schemas found:")
    for table, columns in relevant_schemas.items():
        print(f"Table: {table}, Columns: {columns}")
    
    # Test 6: Performance test
    print("\nTest 6: Performance test")
    start_time = datetime.now()
    for _ in range(10):
        vector_store.get_similar_sql("Find users over 25")
    end_time = datetime.now()
    avg_time = (end_time - start_time).total_seconds() / 10
    print(f"Average query time (10 runs): {avg_time:.3f} seconds")

if __name__ == "__main__":
    main()
