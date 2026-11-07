import json
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]

def data_root():
    value = os.environ.get('MICROBURBS_DATA_ROOT')
    local = PROJECT / 'config.local.json'
    if not value and local.exists():
        value = json.loads(local.read_text()).get('data_root')
    if not value:
        raise RuntimeError('Set MICROBURBS_DATA_ROOT or config.local.json data_root')
    root = Path(value).expanduser().resolve()
    if root == PROJECT or PROJECT in root.parents:
        raise ValueError('Persistent data must be outside the code repository')
    if not root.is_dir():
        raise FileNotFoundError(f'Data mount unavailable: {root}')
    return root

def dump_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str, allow_nan=False)+'\n')
