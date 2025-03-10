from crewai import Tool
import psycopg2
import yaml

class DBTools:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['database']

    def verify_columns(self, sql):
        """Tool to verify if columns in SQL exist in the database."""
        conn = psycopg2.connect(**self.config)
        try:
            cursor = conn.cursor()
            cursor.execute("EXPLAIN " + sql)
            return True, []
        except psycopg2.Error as e:
            return False, str(e)
        finally:
            conn.close()

def get_db_tool(config_path):
    db_tools = DBTools(config_path)
    return Tool(
        name="DBColumnVerifier",
        func=db_tools.verify_columns,
        description="Verifies if columns in a SQL query exist in the database."
    )