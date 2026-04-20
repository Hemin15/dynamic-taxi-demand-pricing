import os

def print_structure(root_dir, indent=""):
    try:
        items = sorted(os.listdir(root_dir))
    except PermissionError:
        return

    for i, item in enumerate(items):
        path = os.path.join(root_dir, item)
        is_last = i == len(items) - 1

        # Tree formatting
        connector = "└── " if is_last else "├── "
        print(indent + connector + item)

        if os.path.isdir(path):
            extension = "    " if is_last else "│   "
            print_structure(path, indent + extension)


if __name__ == "__main__":
    root = "."   # current project folder
    print("\n📁 Project Structure:\n")
    print(root)
    print_structure(root)