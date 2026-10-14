import json

def convert_to_ragas_format(input_file: str, output_file: str):
    with open(input_file, "r") as f:
        data = json.load(f)

    cleaned_data = []

    for item in data:
        # Filter out unanswerable queries
        if not item.get("answerable", True):
            continue

        cleaned_item = {
            "id": item.get("id"),
            # 1. Map 'question' -> 'user_input'
            "user_input": item.get("question"),
            
            # 2. Extract actual passage text (Must be List[str], NOT page numbers)
            "retrieved_contexts": item.get("retrieved_contexts", ["Context text placeholder from vector DB"]),
            
            # 3. Output from your RAG pipeline model
            "response": item.get("response", "Model answer placeholder"),
            
            # 4. Map 'expected_answer' -> 'reference'
            "reference": item.get("expected_answer") or item.get("reference"),
            
            # Additional metadata fields are safely ignored by Ragas
            "question_type": item.get("question_type"),
            "source_reference": item.get("source_reference"),
            "source_page": item.get("source_page")
        }
        cleaned_data.append(cleaned_item)

    with open(output_file, "w") as f:
        json.dump(cleaned_data, f, indent=2)

    print(f"Successfully converted {len(cleaned_data)} items to Ragas format in '{output_file}'.")

# Usage
convert_to_ragas_format("evaluation_questions.json", "ragas_dataset_cleaned.json")