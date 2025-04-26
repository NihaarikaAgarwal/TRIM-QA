import os
import pandas as pd

def calculate_recall_from_folder(folder_path, top_n_list, output_file, percentage_output_file):
    recall_data = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            csv_file = os.path.join(folder_path, filename)
            
            df = pd.read_csv(csv_file)

            print(f"Columns in {filename}: {df.columns.tolist()}")

            for top_n in top_n_list:
                
                row = df.iloc[0]  
                target_table = row['target_table']
                
                top_n_tables = df.head(top_n)['table_id'].tolist()

                recall = 1 if target_table in top_n_tables else 0

                recall_data.append({
                    'query': row['query'],
                    'target_table': target_table,
                    'top_n': top_n,
                    'recall': recall
                })
    
    recall_df = pd.DataFrame(recall_data)

    recall_percentage_data = []
    for top_n in top_n_list:
        total_queries = len(recall_df[recall_df['top_n'] == top_n])
        recall_queries = len(recall_df[(recall_df['top_n'] == top_n) & (recall_df['recall'] == 1)])
        recall_percentage = (recall_queries / total_queries) * 100 if total_queries > 0 else 0
        recall_percentage_data.append({
            'top_n': top_n,
            'recall_percentage': round(recall_percentage, 2)
        })
    recall_percentage_df = pd.DataFrame(recall_percentage_data)

    recall_df.to_csv(output_file, index=False)
    print(f"Recall results saved to {output_file}")

    recall_percentage_df.to_csv(percentage_output_file, index=False)
    print(f"Recall percentages saved to {percentage_output_file}")

    return recall_percentage_df


folder_path = '/home/nagarw48/Projects/TRIM-QA/TRIM-QA/reranked_output'  
top_n_list = [5000, 2500, 1250, 625, 312, 156, 78, 39, 18, 10]  
output_file = 'rerank_recall_results.csv'  
percentage_output_file = 'rerank_recall_percentage_results.csv'  

recall_percentage_df = calculate_recall_from_folder(folder_path, top_n_list, output_file, percentage_output_file)
