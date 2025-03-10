from crewai import Task

class SQLTasks:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def generate_sql_task(self, agent, question):
        similar_sql = self.vector_store.get_similar_sql(question)
        schemas = self.vector_store.get_relevant_schema(question)
        return Task(
            description=f"""Convert this question to SQL: {question}
            Use these similar queries as reference: {similar_sql}
            Available schemas: {schemas}""",
            agent=agent,
            expected_output="A valid SQL query string"
        )

    def validate_sql_task(self, agent, sql):
        return Task(
            description=f"""Validate this SQL query: {sql}
            Use the DBColumnVerifier tool to check if columns and tables exist in the database.
            Return the SQL if valid, or suggest corrections if invalid.""",
            agent=agent,
            expected_output="Validated SQL query or correction suggestions"
        )