"""Small, simulation-independent provenance utilities for bounded RL trials."""
import hashlib
import json
from pathlib import Path
import subprocess


def code_identity():
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()
    paths = git("ls-files", "--cached", "--others", "--exclude-standard").splitlines()
    hashes = {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sorted(set(paths))
              if Path(p).suffix in (".py", ".yaml", ".yml", ".toml", ".sh") and (root/p).is_file()}
    return dict(commit=git("rev-parse", "HEAD"), branch=git("branch", "--show-current"),
                working_tree=git("status", "--short"), source_sha256=hashes,
                source_manifest_sha256=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest())


def write_json(path, value):
    """Atomic progress publication inside a newly created experiment directory."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".partial")
    temporary.write_text(json.dumps(value, indent=2, default=str)+"\n")
    temporary.replace(path)
