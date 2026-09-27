"""Export the installed registry for review; does not change node registration.

Run with the ComfyUI Python environment:
    python scripts/export_node_catalog.py --comfyui-root /path/to/ComfyUI
"""
import argparse
from collections import Counter
import inspect
import json
from pathlib import Path

from migrate_qwen2511_workflow import ROOT, load_nodes

TIER_NAMES={'primary':'主入口','advanced':'专用工具','legacy':'兼容入口','experimental':'实验功能'}


def export(registry, output):
    rows = []
    manifest={row["id"]:row for row in registry.registry.MANIFEST}
    for node_id, cls in registry.NODE_CLASS_MAPPINGS.items():
        category = getattr(cls, 'CATEGORY', '')
        meta=manifest[node_id]
        tier=TIER_NAMES[meta["tier"]]
        schema = cls.INPUT_TYPES()
        rows.append({
            'id': node_id,
            'display': registry.NODE_DISPLAY_NAME_MAPPINGS.get(node_id, node_id),
            'source': str(Path(inspect.getsourcefile(cls)).relative_to(ROOT)),
            'line': inspect.getsourcelines(cls)[1],
            'category': category,
            'proposed_tier': tier,
            'tier':meta['tier'], 'origin':meta['origin'], 'license':meta['license'],
            'legacy_ids':meta['legacy_ids'], 'replacement':meta['replacement'],
            'description':getattr(cls,'DESCRIPTION',''),
            'schema':schema,
            'input_types': {group: {name: data if isinstance(data, str) else data[0] for name, data in values.items()}
                            for group, values in schema.items()},
            'output_types': list(getattr(cls, 'RETURN_TYPES', [])),
            'output_is_list': list(getattr(cls, 'OUTPUT_IS_LIST', [])),
            'input_is_list': bool(getattr(cls, 'INPUT_IS_LIST', False)),
        })
    output.mkdir(parents=True, exist_ok=True)
    (output / 'node-catalog.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
    lines = [
        '# 节点全量清单', '',
        '由唯一注册清单和实际接口生成；层级已应用到菜单。', '',
        f"当前注册 **{len(rows)}** 个节点，**{len(set(r['category'] for r in rows))}** 个分类。", '',
        '| 层级 | 数量 |', '| --- | --- |',
        *[f'| {tier} | {count} |' for tier, count in Counter(r['proposed_tier'] for r in rows).items()], '',
        '| 层级 | 节点 ID | 显示名 | 当前分类 | 输出 | 源码 |',
        '| --- | --- | --- | --- | --- | --- |',
    ]
    for row in rows:
        outputs = ', '.join(row['output_types']) or '输出节点'
        lines.append(f"| {row['proposed_tier']} | `{row['id']}` | {row['display']} | {row['category']} | {outputs} | [{row['source']}](../{row['source']}#L{row['line']}) |")
    lines += ['', '机器可读接口清单：[node-catalog.json](node-catalog.json)。', '',
              '接口类型相似不代表行为等价；合并前还需比较批次、空输入、遮罩约定、设备及像素结果。', '']
    lines += ['## 历史 ID 和替代入口', '', '| 历史 ID | 当前独立 ID | 推荐替代 |', '| --- | --- | --- |']
    for row in rows:
        if row['legacy_ids'] or row['replacement']:
            lines.append(f"| {', '.join(row['legacy_ids']) or '—'} | `{row['id']}` | {row['replacement'] or '保留当前入口'} |")
    lines.append('')
    (output / 'NODE_CATALOG.md').write_text('\n'.join(lines))
    print(f'Exported {len(rows)} nodes to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--comfyui-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'docs')
    options = parser.parse_args()
    export(load_nodes(options.comfyui_root), options.output)
