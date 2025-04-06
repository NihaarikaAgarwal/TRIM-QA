import os
import json
import random
def generate_random_queries(file_paths, queries_count, saved_queries_path):
    # Check if the random queries already exist
    if os.path.exists(saved_queries_path):
        print(f"Loading saved queries from {saved_queries_path}")
        with open(saved_queries_path, 'r') as f:
            selected_queries = [json.loads(line) for line in f]
    else:
        # Combine all queries from the three files
        all_queries = []
        for file_path in file_paths:
            with open(file_path, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    all_queries.append(data)
        
        # Shuffle queries and select the random queries_count
        random.shuffle(all_queries)
        selected_queries = all_queries[:queries_count]
        
        # Save the random queries to a JSONL file for future use
        print(f"Saving random queries to {saved_queries_path}")
        with open(saved_queries_path, 'w') as f:
            for query in selected_queries:
                json.dump(query, f)
                f.write("\n")
#main
file_paths = ["test.jsonl", "train.jsonl", "dev.jsonl"]
queries_count = 150
saved_queries_path = "saved_random_queries.jsonl"
generate_random_queries(file_paths, queries_count, saved_queries_path)