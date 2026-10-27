import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
let extension;
const source=fs.readFileSync(new URL('../web/js/qwen21_controls.js',import.meta.url),'utf8').replace(/^import .*;\n/gm,'');
vm.runInNewContext(source,{app:{registerExtension:e=>extension=e},api:{},Set,Math});
class Node {
 constructor(){this.size=[420,500];this.widgets=['task','extra_prompt','lora_strength','seed','resolution','sampling_preset','consistency','people','swap_head','forehead','bfs_version','skin_match','watermark_mode','left','right','top','bottom','azimuth','elevation','close_up'].map(name=>({name,type:'number',value:name==='task'?'remove':name==='lora_strength'?.48:name==='extra_prompt'?'保留肤色; smile':1,computeSize:()=>[420,20]}));this.onNodeCreated();}
 setSize(size){this.size=size;}computeSize(){return [420,300];}setDirtyCanvas(){}
}
await extension.beforeRegisterNodeDef(Node,{name:'SwwanQwen21Controls'});
const n=new Node();const widget=name=>n.widgets.find(w=>w.name===name);
for(const strength of [0,.48,1]){
 widget('lora_strength').value=strength;
 for(const task of ['face_swap','outpaint','edit','local_edit','angles','remove']){
  widget('task').value=task;widget('task').callback(task);
  assert.equal(widget('lora_strength').value,strength);
  assert.equal(widget('lora_strength').type,'number');
  assert.equal(widget('extra_prompt').value,'保留肤色; smile');
  assert.equal(widget('people').type==='number',task==='face_swap');
  assert.equal(widget('consistency').type==='number',['edit','local_edit'].includes(task));
  const values=JSON.parse(JSON.stringify(n.widgets.map(w=>w.value)));
  const reopened=new Node();reopened.widgets.forEach((w,i)=>w.value=values[i]);reopened.onConfigure();
  assert.equal(reopened.widgets.find(w=>w.name==='lora_strength').value,strength);
 }
}
console.log('PASS: task switches, absolute strength 0/0.48/1, prompt preservation and save/reload values');
