config_json = """
{
    "model_name": "sentence-transformers/all-mpnet-base-v2",
    "gemini_api_key": "YOUR_API_KEY",
    "knowledge_base_file": "knowledge_base.xlsx",
    "faiss_index_file": "knowledge_base.index",
    "processed_data_file": "processed_data.json",
    "keyword_column": "Keyword",
    "description_column": "Description",
    "command_column": "Function",
    "sheet_name": "key_des_func",
    "result_commands_file": "result/result_commands.txt",
    "split_prompt_file": "result/split_prompt_file.txt",
    "split_result_file": "result/split_result_file.txt",
    "generate_prompt_file": "result/generate_prompt_file.txt",
    "check_file": "result/check_file.txt",
    "pseudo_code_file": "result/pseudo_code_file.txt",
    "valid_pseudo_code_file": "result/valid_pseudo_code_file.txt",
    "search_result_file": "result/search_result.txt",
    "keyword_index_file": "keyword_index.txt",
    "index_map_file": "index_map.json",
    "preprocessing": {
        "lowercase": false,
        "strip": true
    },
    "search_top_k": 2,
    "max_attemps": 2,
    "gemini_model": "gemini-2.0-flash-001",
    "search_model": "gemini-2.0-flash-001"
}
"""

with open('RAG/config.json', 'w', encoding='utf-8') as f:
    f.write(config_json)

print("The config.json file has been created and written.")
