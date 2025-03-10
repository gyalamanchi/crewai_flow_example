from flows.sql_flow import SQLGenerationFlow

def main():
    # Get question from user input
    question = input("Enter your question to convert to SQL: ")
    
    # Initialize and run the flow
    config_path = "config/db_config.yaml"
    flow = SQLGenerationFlow(config_path=config_path, question=question)
    
    # Execute the flow
    result = flow.run()
    
    return result

if __name__ == "__main__":
    main()