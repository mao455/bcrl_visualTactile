"""Read retained evidence or an explicitly retired, hashed historical Git blob."""
import hashlib
import json
from pathlib import Path
import subprocess


def payload(root, name):
    root=Path(root)
    path=root/name
    if path.is_file():
        return path.read_bytes()
    registry=json.loads((root/'retired_source_paths.json').read_text())
    row=registry[name]
    project=root.parents[2]
    value=subprocess.check_output(['git','-C',str(project),'cat-file','blob',row['git_blob']])
    if hashlib.sha256(value).hexdigest()!=row['sha256']:
        raise ValueError('Retired historical source identity differs: '+name)
    return value
