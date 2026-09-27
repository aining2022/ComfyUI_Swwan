"""Read the generated catalog; ranking is discovery, not semantic equivalence."""
import argparse
import json
from pathlib import Path

ALIASES={'缩放':['resize','scale'],'裁剪':['crop'],'还原':['restore','uncrop'],
         '保存':['save'],'透明':['rgba','alpha'],'批次':['batch','list'],
         '遮罩':['mask'],'颜色':['color','rgb'],'拼接':['concat','grid'],'数学':['math','expression']}

def candidates(query,repo,limit=8):
    rows=json.loads((Path(repo)/'docs/node-catalog.json').read_text())
    terms=query.lower().split()
    for word,aliases in ALIASES.items():
        if word in query:terms.extend(aliases)
    result=[]
    for row in rows:
        text=(row['id']+' '+row['display']+' '+row.get('description','')+' '+json.dumps(row.get('schema',{}),ensure_ascii=False)).lower()
        score=sum(5 if term in row['id'].lower() else 1 for term in terms if term in text)
        if score:result.append((score+(2 if row.get('tier')=='primary' else 0),row))
    return [row for _,row in sorted(result,key=lambda item:(-item[0],item[1]['id']))[:limit]]

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('query');parser.add_argument('--repo',type=Path);parser.add_argument('--limit',type=int,default=8)
    args=parser.parse_args();repo=args.repo
    if repo is None:
        options=[Path.cwd(),*Path.cwd().parents,Path(__file__).resolve().parents[3]]
        repo=next((p for p in options if (p/'node_manifest.json').exists()),None)
    if repo is None:parser.error('Specify --repo; no ComfyUI_Swwan checkout found.')
    for row in candidates(args.query,repo,args.limit):
        print(json.dumps({k:row.get(k) for k in ['id','tier','source','input_types','output_types','replacement']},ensure_ascii=False))
