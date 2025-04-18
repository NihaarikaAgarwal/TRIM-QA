import os
import pandas as pd

# === CONFIGURATION ===
FOLDER_PATH = "/home/nagarw48/Projects/TRIM-QA/TRIM-QA/output_archive"
X_LIST = [5000, 2500, 1250, 625, 312, 156, 78, 39, 18, 10]
OUTPUT_CSV = "aggregated_reranked_recall_results_faiss.csv"

def calculate_recall_for_csv(file_path, top_k_10, top_k_20):
    df = pd.read_csv(file_path)

    # Strip whitespace and normalize to lowercase
    df.columns = [col.strip().lower() for col in df.columns]
    
    #print(f"[DEBUG] Normalized columns in {file_path}: {df.columns.tolist()}")

    # Now use normalized names
    target_table = df.iloc[0]["target_table"] if "target_table" in df.columns else df.iloc[0][3]

    if "similarity_score" not in df.columns:
        raise KeyError(f"'similarity_score' column not found in {file_path}. Check for typos or format issues.")

    df_sorted = df.sort_values(by="similarity_score", ascending=False).reset_index(drop=True)

    top_10_ids = df_sorted.iloc[:top_k_10]["table_id"].tolist()
    top_20_ids = df_sorted.iloc[:top_k_20]["table_id"].tolist()

    recall_10 = 1 if target_table in top_10_ids else 0
    recall_20 = 1 if target_table in top_20_ids else 0
    return recall_10, recall_20


def compute_aggregated_recall(folder_path, x_list):
    all_results = []

    for x in x_list:
        top_k_10 = int(x * 0.10)
        top_k_20 = int(x * 0.20)

        recall_10_list = []
        recall_20_list = []

        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(folder_path, filename)
                recall_10, recall_20 = calculate_recall_for_csv(file_path, top_k_10, top_k_20)
                recall_10_list.append(recall_10)
                recall_20_list.append(recall_20)

        recall_10_percent = 100 * sum(recall_10_list) / len(recall_10_list)
        recall_20_percent = 100 * sum(recall_20_list) / len(recall_20_list)

        all_results.append({
            "selected_top_x": x,
            "recall@10%": round(recall_10_percent, 2),
            "recall@20%": round(recall_20_percent, 2)
        })

    result_df = pd.DataFrame(all_results)
    result_df.to_csv(OUTPUT_CSV, index=False)
    print("Saved recall results for all X values to:", OUTPUT_CSV)
    return result_df

compute_aggregated_recall(FOLDER_PATH, X_LIST)
