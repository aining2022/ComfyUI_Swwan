"""Generate model-free UI examples from current node contracts."""
import argparse
import asyncio
import json
from pathlib import Path
from migrate_qwen2511_workflow import ROOT,load_nodes,defaults
p=argparse.ArgumentParser();p.add_argument('--comfyui-root',type=Path,required=True);args=p.parse_args()
reg=load_nodes(args.comfyui_root)
import nodes
asyncio.run(nodes.load_custom_node(str(args.comfyui_root/'comfy_extras/nodes_mask.py'),module_parent='comfy_extras'))
classes={**nodes.NODE_CLASS_MAPPINGS,**reg.NODE_CLASS_MAPPINGS}
SCALARS={'INT','FLOAT','BOOLEAN','STRING','COLORCODE'}
def graph(specs,connections,name):
    graphnodes=[]
    for num,(kind,params) in enumerate(specs,1):
        cls=classes[kind];schema=cls.INPUT_TYPES();values=defaults(cls);values.update(params)
        inputs=[]
        for group in ['required','optional']:
            for field,data in schema.get(group,{}).items():
                typ=data[0];scalar=isinstance(typ,list) or typ in SCALARS
                port={'name':field,'type':'COMBO' if isinstance(typ,list) else typ,'link':None}
                if scalar:port['widget']={'name':field}
                inputs.append(port)
        if kind=='SwwanImageConcatMulti':
            for index in range(3,int(values['inputcount'])+1):inputs.append({'name':f'image_{index}','type':'IMAGE','link':None})
        graphnodes.append({'id':num,'type':kind,'pos':[30+(num-1)%3*360,30+(num-1)//3*440],
          'size':[330,320],'flags':{},'order':num-1,'mode':0,'inputs':inputs,
          'outputs':[{'name':getattr(cls,'RETURN_NAMES',cls.RETURN_TYPES)[i],'type':t,'links':None} for i,t in enumerate(cls.RETURN_TYPES)],
          'properties':{'Node name for S&R':kind,'swwan_version':'1.0.0'} if kind in reg.NODE_CLASS_MAPPINGS else {'Node name for S&R':kind},
          'widgets_values':[values[key] for key in defaults(cls)]})
    links=[]
    for index,(source,slot,target,field) in enumerate(connections,1):
        a=graphnodes[source-1];b=graphnodes[target-1];port=next(i for i,item in enumerate(b['inputs']) if item['name']==field)
        b['inputs'][port]['link']=index;a['outputs'][slot]['links']=(a['outputs'][slot]['links'] or [])+[index]
        links.append([index,source,slot,target,port,a['outputs'][slot]['type']])
    result={'last_node_id':len(graphnodes),'last_link_id':len(links),'nodes':graphnodes,'links':links,'groups':[],
            'config':{},'extra':{'ds':{'scale':.8,'offset':[20,20]}},'version':.4}
    (ROOT/'examples'/name).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
graph([
 ('LayerUtility: ColorImage (Swwan)',{'width':128,'height':96,'color':'#364254'}),
 ('LayerUtility: ColorImage (Swwan)',{'width':128,'height':96,'color':'#d08a53'}),
 ('ImageResizeKJv2Alternative',{'resize_mode':'aspect_ratio','aspect_ratio':'1:1','aspect_length':128}),
 ('SwwanImageConcatMulti',{'layout':'grid','columns':2,'match_image_size':True,'inputcount':4}),
 ('SwwanImagesToRGB',{}),
 ('SwwanSaveImage',{'output_path':'swwan_examples','filename_prefix':'image_tools','save_workflow_as_json':True}),
],[(1,0,3,'image'),(3,0,4,'image_1'),(2,0,4,'image_2'),(1,0,4,'image_3'),(2,0,4,'image_4'),(4,0,5,'images'),(5,0,6,'image')],'cpu-image-tools.json')
graph([
 ('LayerUtility: ColorImage (Swwan)',{'width':128,'height':96,'color':'#237bbd'}),
 ('SolidMask',{'width':128,'height':96,'value':.5}),
 ('SwwanSaveImage',{'output_path':'swwan_examples','filename_prefix':'rgba','file_format':'png','alpha_mode':'keep','save_workflow_as_json':True}),
],[(1,0,3,'image'),(2,0,3,'alpha')],'cpu-rgba.json')
print('Generated two CPU examples')
