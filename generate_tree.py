import os

EXCLUDED_DIRS = {'.venv', '__pycache__', '.git'}

def print_directory_tree(startpath, prefix=''):
    items = [item for item in os.listdir(startpath) if item not in EXCLUDED_DIRS]
    entries = sorted(items, key=lambda s: s.lower())
    for i, entry in enumerate(entries):
        path = os.path.join(startpath, entry)
        connector = "├── " if i < len(entries) - 1 else "└── "
        print(prefix + connector + entry)
        if os.path.isdir(path):
            extension = "│   " if i < len(entries) - 1 else "    "
            print_directory_tree(path, prefix + extension)

if __name__ == "__main__":
    root_dir = "."  # or set to a specific folder
    print(f"Directory tree for: {os.path.abspath(root_dir)}\n")
    print_directory_tree(root_dir)
