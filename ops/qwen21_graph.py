# SPDX-License-Identifier: MIT
"""Adapt the user's frozen BFS v5 graph, preserving segmentation and freezing row-major regions."""
import json
from pathlib import Path


def face_graph(model,clip,vae,image,mask,references,config):
    path=Path(__file__).resolve().parents[1]/'resources/qwen21/bfs-v5-api.json'
    graph=json.loads(path.read_text())
    def node(kind,**inputs):return {'class_type':kind,'inputs':inputs}
    graph['140']=node('SwwanQwen21Source',image=image,mask=mask)
    graph['233']=node('SwwanQwen21FaceRegions',segs=['124',0],index=-1)
    graph['110']=node('SwwanQwen21FaceRegions',segs=['233',0],index=['72',1])
    graph['202']=node('SwwanQwen21ReferenceAtIndex',references=references,index=['72',1])
    graph['225']['inputs']['值']=config['people']
    graph['194']['inputs']['开关']=config['swap_head']
    graph['195']['inputs']['开关']=config['forehead']
    graph['272']['inputs']['value']=config['skin_match']
    graph['226']=node('SwwanQwen21FaceGuard',image=['140',0],count=['117',0],valid=['224',7],references=references,config=config)
    graph['checkpoint']=node('SwwanQwen21FaceCheckpoint',image=['72',2],person=['72',1],config=config)
    graph['131']['inputs'].update(crop_mode='qwen21_aligned',qwen_resolution=config['resolution'],qwen_tolerance=1.,image=['checkpoint',0])
    graph['reference']=node('SwwanQwen21Auxiliary',image=['13',0])
    graph['247']=node('SwwanQwen21Core',model=model,clip=clip,vae=vae,image=['131',0],
                      references=['reference',0],config=config,person=['72',1],previous=['checkpoint',0])
    graph['63']=node('SwwanQwen21FaceRestore',background=['72',2],image=['285',0],seam=['131',4],
                    mask=['133',0],manual=['129',7],config=config,record=['247',1],person=['72',1])
    graph['final']=node('SwwanQwen21FaceFinish',image=['162',0],preview=['230',0],config=config)
    replacements={('138',0):['131',0],('138',1):['131',6],('138',2):['131',7],('138',3):['131',5],('249',0):['247',0]}
    for item in graph.values():
        for k,v in list(item['inputs'].items()):
            if isinstance(v,list) and len(v)==2 and isinstance(v[0],str) and isinstance(v[1],int):
                item['inputs'][k]=replacements.get(tuple(v),v)
    # Only ancestors of the active result run. No old loader, cache, sampler or unused preview remains.
    used=set()
    def visit(k):
        if k in used:return
        used.add(k)
        for v in graph[k]['inputs'].values():
            if isinstance(v,list) and len(v)==2 and isinstance(v[0],str) and isinstance(v[1],int):visit(v[0])
    visit('final')
    return {k:v for k,v in graph.items() if k in used}


def dependencies(graph, outputs):
    used=set()
    def visit(k):
        if k in used:return
        used.add(k)
        for value in graph[k]['inputs'].values():
            if isinstance(value,list) and len(value)==2 and isinstance(value[0],str) and isinstance(value[1],int):visit(value[0])
    for k in outputs:visit(k)
    return {k:v for k,v in graph.items() if k in used}


def build_face_graph(model,clip,vae,image,mask,references,config):
    from comfy_execution.graph_utils import GraphBuilder,add_graph_prefix
    graph=face_graph(model,clip,vae,image,mask,references,config)
    graph['sequence']={'class_type':'SwwanQwen21FaceSequence','inputs':dict(
        model=model,clip=clip,vae=vae,image=['226',0],mask=mask,regions=['233',0],
        manual=['129',7],group_threshold=0.,count=['117',0],preview=['230',0],
        references=references,config=config)}
    graph=dependencies(graph,['sequence'])
    graph,outputs=add_graph_prefix(graph,(['sequence',0],['sequence',1]),GraphBuilder.alloc_prefix())
    return {'result':outputs,'expand':graph}


def build_face_sequence(model,clip,vae,image,mask,regions,manual,group_threshold,count,preview,references,config):
    from comfy_execution.graph_utils import GraphBuilder,add_graph_prefix
    # Regions and order are already evaluated on the original image. Unroll only the real count.
    # Easy-Use's nested loop expansion loses node IDs inside another dynamic expansion.
    expanded={};previous=image
    for index in range(count):
        graph=face_graph(model,clip,vae,image,mask,references,config)
        constants={('72',1):index,('72',2):previous,('124',0):regions,('233',0):regions,
                   ('129',7):manual,('208',0):group_threshold}
        for item in graph.values():
            for key,value in list(item['inputs'].items()):
                if isinstance(value,list) and len(value)==2 and isinstance(value[0],str) and isinstance(value[1],int):
                    if tuple(value) in constants:item['inputs'][key]=constants[tuple(value)]
        # The previous person's canvas is an external link. Hide it from ancestry/prefix rewriting.
        for node_id in ('checkpoint','63','91'):
            field={'checkpoint':'image','63':'background','91':'image'}[node_id]
            if node_id in graph:graph[node_id]['inputs'][field]=image
        graph=dependencies(graph,['63'])
        prefix=GraphBuilder.alloc_prefix()
        graph,outputs=add_graph_prefix(graph,(['63',0],),prefix)
        for node_id,field in (('checkpoint','image'),('63','background'),('91','image')):
            if prefix+node_id in graph:graph[prefix+node_id]['inputs'][field]=previous
        expanded.update(graph);previous=outputs[0]
    graph=GraphBuilder();done=graph.node('SwwanQwen21FaceFinish',image=previous,preview=preview,config=config)
    expanded.update(graph.finalize())
    return {'result':(done.out(0),done.out(1)),'expand':expanded}


def row_order(segs):
    """Top-to-bottom rows, left-to-right within each row; only the original SEGS are sorted."""
    import statistics
    shape,items=segs
    if not items:return shape,[]
    def center(seg):
        x0,y0,x1,y1=seg.bbox
        return (y0+y1)/2,(x0+x1)/2
    tolerance=max(1.,statistics.median(max(1,s.bbox[3]-s.bbox[1]) for s in items)*.6)
    rows=[]
    for seg in sorted(items,key=center):
        cy,cx=center(seg)
        if rows and abs(cy-statistics.mean(center(s)[0] for s in rows[-1]))<=tolerance:
            rows[-1].append(seg)
        else:rows.append([seg])
    return shape,[seg for row in rows for seg in sorted(row,key=lambda s:(center(s)[1],center(s)[0]))]
