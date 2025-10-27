
import json
import requests
import time

API_URL = "http://127.0.0.1:8000/api/v1/chat"
EVAL_FILE_PATH = "eval/gold_truth.jsonl"

def run_evaluation():
    """
    Loads an evaluation dataset, runs each item against the chat API,
    and calculates retrieval precision.
    """
    test_cases = []
    with open(EVAL_FILE_PATH, 'r') as f:
        for line in f:
            test_cases.append(json.loads(line))

    print(f"Loaded {len(test_cases)} test cases.")
    
    retrieval_hits = 0
    total_retrieval_queries = 0
    
    for i, case in enumerate(test_cases):
        question = case['question']
        expected_sources = set(case['source_doc_ids'])
        
        print(f"\n--- Running Test Case {i+1}/{len(test_cases)} ---")
        print(f"Question: {question}")
        
        # call chat api
        response = requests.post(API_URL, json={"query": question})
        
        if response.status_code != 200:
            print(f"  ERROR: API returned status {response.status_code}")
            print(f"  Response: {response.text}")
            continue

        response_data = response.json()
        generated_answer = response_data.get('answer', 'N/A')
        retrieved_ids = set(response_data.get('retrieved_ids', []))

        print(f"  Ground Truth Answer: {case['ground_truth_answer']}")
        print(f"  Generated Answer: {generated_answer}")

        # Calculate retrieval precision
        if expected_sources:
            total_retrieval_queries += 1
            is_hit = bool(expected_sources.intersection(retrieved_ids))
            if is_hit:
                retrieval_hits += 1
            print(f"  Retrieval Check: {'HIT' if is_hit else 'MISS'}")
            print(f"    - Expected: {expected_sources}")
            print(f"    - Retrieved: {retrieved_ids}")

    print("\n--- Evaluation Summary ---")
    if total_retrieval_queries > 0:
        precision = (retrieval_hits / total_retrieval_queries) * 100
        print(f"Retrieval Precision: {retrieval_hits}/{total_retrieval_queries} = {precision:.2f}%")
    else:
        print("No queries with expected sources were run.")
    print("--------------------------")

if __name__ == "__main__":
    run_evaluation()