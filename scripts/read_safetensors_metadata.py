import sys
from safetensors import safe_open
import json

def get_metadata(path):
    print(f"Reading metadata from {path}...")
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
            if not metadata:
                print("No metadata found.")
                return

            config_str = metadata.get("config")
            if config_str:
                try:
                    config = json.loads(config_str)
                    print(json.dumps(config, indent=2))
                except json.JSONDecodeError:
                    print(f"Raw config string: {config_str}")
            else:
                print("No 'config' key in metadata.")
                print(f"All metadata keys: {list(metadata.keys())}")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python read_safetensors_metadata.py <path_to_safetensors>")
    else:
        get_metadata(sys.argv[1])
