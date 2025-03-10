from crewai import Flow, step, start, listen
from agents.sql_agents import SQLAgents
from tasks.sql_tasks import SQLTasks
from embeddings.vector_store import VectorStore

class SQLGenerationFlow(Flow):
    def __init__(self, config_path, question):
        super().__init__()
        self.config_path = config_path
        self.question = question
        self.vector_store = VectorStore(config_path)
        self.agents = SQLAgents(self.vector_store, config_path)
        self.tasks = SQLTasks(self.vector_store)
        self.generated_sql = None

    @start()
    def initialize(self):
        print(f"Starting SQL generation for question: {self.question}")
        return {"question": self.question}

    @step()
    def generate_sql(self, state):
        agent = self.agents.get_sql_generator()
        task = self.tasks.generate_sql_task(agent, self.question)
        self.generated_sql = task.execute()
        return {"sql": self.generated_sql}

    @step()
    @listen("generate_sql")
    def validate_sql(self, state):
        agent = self.agents.get_sql_validator()
        task = self.tasks.validate_sql_task(agent, state["sql"])
        validation_result = task.execute()
        is_valid, errors = agent.tools[0].func(state["sql"])  # Use the tool directly for verification
        return {
            "sql": state["sql"],
            "is_valid": is_valid,
            "validation_result": validation_result,
            "errors": errors
        }

    @step()
    @listen("validate_sql")
    def present_results(self, state):
        print("\nResults:")
        print(f"Generated SQL: {state['sql']}")
        if state['is_valid']:
            print("SQL is valid!")
        else:
            print(f"SQL validation errors: {state['errors']}")
            print(f"Validation feedback: {state['validation_result']}")
        return state