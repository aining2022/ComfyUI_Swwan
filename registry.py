"""Single public registry. IDs and menu metadata come only from node_manifest.json."""
import importlib
import json
from pathlib import Path

MANIFEST = json.loads((Path(__file__).with_name('node_manifest.json')).read_text())


def build_registry(package, manifest=MANIFEST):
    classes, names = {}, {}
    for entry in manifest:
        node_id = entry['id']
        if node_id in classes:
            raise ValueError(f'Duplicate Swwan node ID: {node_id}')
        cls = getattr(importlib.import_module('.'+entry['module'], package), entry['class_name'])
        cls.CATEGORY = entry['category']
        description = getattr(cls, 'DESCRIPTION', '')
        if entry.get('replacement'):
            note = f"旧版兼容；推荐入口：{entry['replacement']}。保留历史数据合同。"
            if note not in description:
                description += "\n" + note
        cls.DESCRIPTION = description
        classes[node_id] = cls
        names[node_id] = entry['display']
    return classes, names
