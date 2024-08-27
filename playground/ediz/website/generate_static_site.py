import os
from pathlib import Path
from flask import render_template
import shutil
from jinja2 import Environment, FileSystemLoader

# Configuration
app_root = Path(__file__).parent.resolve()
selection_root = app_root / "../selection"
output_dir = app_root / "_static_site"
templates_dir = app_root / "templates"
static_dir = app_root / "static"

# Setup Jinja2 environment
env = Environment(loader=FileSystemLoader(templates_dir))

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

def render_static_site(root_structure, subdir_structure):
    """
    Renders the static site into the output directory.
    """
    # Create the output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Render the index page at the root level
    template = env.get_template('index.html')
    index_content = template.render(root_structure=root_structure, subdir_structure=subdir_structure, base_path="")
    index_path = output_dir / "index.html"
    with open(index_path, 'w') as f:
        f.write(index_content)

    # Render HTML files in each subdirectory
    for folder, content in subdir_structure.items():
        folder_output_dir = output_dir / folder
        folder_output_dir.mkdir(parents=True, exist_ok=True)

        for file in content['files']:
            file_output_path = folder_output_dir / file
            # Copy the HTML file from the source directory to the output
            shutil.copyfile(selection_root / folder / file, file_output_path)
            
        # Render the index.html in each subdirectory with the correct base_path
        sub_index_content = template.render(root_structure=[], subdir_structure={folder: content}, base_path="../")
        sub_index_path = folder_output_dir / "index.html"
        with open(sub_index_path, 'w') as f:
            f.write(sub_index_content)

    # Copy static assets (CSS, JS, images, etc.)
    if static_dir.exists():
        shutil.copytree(static_dir, output_dir / 'static', dirs_exist_ok=True)

    # Corrected print statement
    print(f"Static website generated at: {output_dir}")

if __name__ == "__main__":
    root_structure, subdir_structure = get_directory_structure(selection_root)
    render_static_site(root_structure, subdir_structure)
