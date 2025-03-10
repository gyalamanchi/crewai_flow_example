from crewai import Agent
from tools.db_tools import get_db_tool

class SQLAgents:
    def __init__(self, vector_store, config_path):
        self.vector_store = vector_store
        self.config_path = config_path

    def get_sql_generator(self):
        return Agent(
            role='SQL Generator',
            goal='Convert natural language to SQL queries',
            backstory='Expert in database query generation and SQL syntax',
            verbose=True,
            tools=[]  # No tools needed for generation
        )

    def get_sql_validator(self):
        return Agent(
            role='SQL Validator',
            goal='Validate SQL queries against database schema',
            backstory='Specialist in database schema verification',
            verbose=True,
            tools=[get_db_tool(self.config_path)]
        )