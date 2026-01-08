import json

notebook_path = "notebooks/candle_svd_test.ipynb"

try:
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Find the compile_project cell
    target_id = "compile_project"
    found = False
    
    for cell in nb["cells"]:
        if cell.get("metadata", {}).get("id") == target_id:
            found = True
            source = cell["source"]
            
            # Check if already added
            has_path = any("LIBRARY_PATH" in line for line in source)
            
            if not has_path:
                # Insert path exports before the cargo build/clean commands
                # We usually look for "cd candle-video" or similar start
                insert_idx = 0
                for i, line in enumerate(source):
                    if "cd candle-video" in line:
                        insert_idx = i + 1
                        break
                
                new_lines = [
                    "\n",
                    "# Добавляем пути к CUDA/cuDNN библиотекам\n",
                    "export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH\n",
                    "export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH\n",
                    "export CUDA_ROOT=/usr/local/cuda\n",
                    "\n"
                ]
                
                # Insert
                for line in reversed(new_lines):
                    source.insert(insert_idx, line)
                
                print(f"Updated paths in cell '{target_id}'")
            else:
                print(f"Paths already present in cell '{target_id}'")
            break
            
    if found:
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=2, ensure_ascii=False)
        print("Notebook updated successfully.")
    else:
        print(f"Cell '{target_id}' not found.")

except Exception as e:
    print(f"Error: {e}")
