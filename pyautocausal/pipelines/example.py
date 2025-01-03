# Example usage, this won't work
def build_pipeline():
    graph = ExecutableGraph()
    
    # Load data
    load_data = ActionNode("load_data", graph, 
        lambda: pd.read_csv("data.csv"))
    
    # Validate data size
    validate = ActionNode("validate", graph,
        lambda data: validate_data(data))
    validate.add_ancestor(load_data)
    
    # Analysis node with condition based on validation
    analysis = ActionNode("analysis", graph,
        lambda data: analyze_data(data))
    analysis.add_ancestor(load_data)
    analysis.add_ancestor(validate)
    
    # Condition that depends on validation output
    analysis.set_condition(
        lambda validate: validate['quality_score'] > 0.9,  # condition function
        ['validate'],  # ancestor node names to pass to condition
        "Data quality too low for analysis"  # skip reason
    )
    
    # More complex condition using multiple ancestors
    summary = ActionNode("summary", graph, 
        lambda data, validate: summarize_data(data, validate))
    summary.add_ancestor(load_data)
    summary.add_ancestor(validate)
    summary.set_condition(
        lambda load_data, validate: (
            len(load_data) > 1000 and 
            validate['quality_score'] > 0.8
        ),
        ['load_data', 'validate'],
        "Dataset too small or quality too low"
    )