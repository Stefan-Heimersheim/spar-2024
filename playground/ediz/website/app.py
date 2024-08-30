from flask import Flask, render_template, send_from_directory, abort
import os
from pathlib import Path

app = Flask(__name__)

# Define the root directory for your 'selection' files
selection_root = Path("../selection").resolve()

def get_directory_structure(rootdir: Path):
    """
    Creates a dictionary for the root and a nested dictionary for subdirectories.
    """
    root_structure = []
    subdir_structure = {}

    for root, dirs, files in os.walk(rootdir):
        folder = os.path.relpath(root, rootdir)
        
        if folder == ".":
            # We're in the root directory
            root_structure = sorted(dirs)
        else:
            # This is a subdirectory
            subdir_structure[folder] = {
                "files": sorted(f for f in files if f.endswith(".html")),
                "subdirs": sorted(dirs)
            }
    
    return root_structure, subdir_structure

@app.route("/")
def index():
    """
    Home route, displays the directory structure.
    """
    root_structure, subdir_structure = get_directory_structure(selection_root)
    return render_template("index.html", root_structure=root_structure, subdir_structure=subdir_structure, base_path="")

@app.route("/<path:file_path>")
def view_file(file_path):
    """
    Route to view an HTML file directly.
    """
    full_path = selection_root / file_path
    if not full_path.exists() or not full_path.is_file():
        abort(404)
    return send_from_directory(full_path.parent, full_path.name)

@app.errorhandler(404)
def page_not_found(e):
    """
    Error handler for 404 pages.
    """
    return render_template('404.html'), 404

if __name__ == "__main__":
    app.run(debug=True)
