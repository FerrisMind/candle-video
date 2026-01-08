import json
import os

notebook_path = "notebooks/candle_svd_test.ipynb"

try:
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Define the new cell
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {
            "id": "install_cudnn"
        },
        "outputs": [],
        "source": [
            "%%bash\n",
            "# Устанавливаем cuDNN (для ускорения сверток)\n",
            "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb\n",
            "dpkg -i cuda-keyring_1.1-1_all.deb\n",
            "apt-get update\n",
            "apt-get -y install cudnn9-cuda-12\n",
            "echo \"✅ cuDNN installed!\""
        ]
    }

    # Find position to insert (after install_rust)
    insert_idx = -1
    for i, cell in enumerate(nb["cells"]):
        if cell.get("metadata", {}).get("id") == "install_rust":
            insert_idx = i + 1
            break
    
    if insert_idx == -1:
        # Fallback: insert at index 2
        insert_idx = 2

    # Check if already exists to avoid duplicates
    exists = False
    for cell in nb["cells"]:
        if cell.get("metadata", {}).get("id") == "install_cudnn":
            exists = True
            break
    
    if not exists:
        nb["cells"].insert(insert_idx, new_cell)
        print(f"Inserted cuDNN cell at index {insert_idx}")
        
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=2, ensure_ascii=False)
        print("Notebook updated successfully.")
    else:
        print("cuDNN cell already exists.")

except Exception as e:
    print(f"Error: {e}")
