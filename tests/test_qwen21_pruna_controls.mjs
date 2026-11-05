// SPDX-License-Identifier: MIT
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
const extensions=[];
vm.runInNewContext(readFileSync(new URL('../web/js/qwen21_pruna_controls.js',import.meta.url),'utf8').replace(/^import .*;\n/gm,''),{app:{registerExtension:e=>extensions.push(e)}});
const workflow=JSON.parse(readFileSync(new URL('../examples/qwen21-pruna-v4.json',import.meta.url),'utf8'));
const def=workflow.nodes.find(n=>n.type==='SwwanQwen21PrunaControls');
class Node {
 constructor(){this.comfyClass=def.type;this.inputs=[];this.widgets=Object.entries(def.widgets_values_named).map(([name,value])=>({name,value,type:'combo',computeSize:()=>[200,20]}));}
 computeSize(){return [500,500];} setSize(size){this.size=size;}
}
extensions[0].beforeRegisterNodeDef(Node,{name:def.type});
const n=new Node(); n.onNodeCreated();
const w=name=>n.widgets.find(w=>w.name===name);
const set=(name,value)=>{w(name).value=value;w(name).callback(value);};
assert.equal(w('left').hidden,true); assert.notEqual(w('task_lora_edit').hidden,true);
set('task','outpaint'); assert.notEqual(w('left').hidden,true);assert.equal(w('task_lora_edit').hidden,true);assert.equal(w('additional_lora_name').hidden,true);
set('task','remove');assert.equal(w('block_size').hidden,true);set('enhanced',true);assert.notEqual(w('block_size').hidden,true);assert.equal(w('mark_opacity').hidden,true);
set('task','face_swap');set('skin_match',true);assert.notEqual(w('skin_strength').hidden,true);assert.equal(w('marking').hidden,true);
set('task','photoreal');set('generation_mode','text');assert.equal(w('qwen_min_resolution').hidden,true);assert.notEqual(w('output_width').hidden,true);
set('task','cutout');assert.equal(w('color_mode').hidden,true);
assert.equal(w('left').value,128);assert.equal(w('skin_strength').value,.25);
w('left').type='converted-widget';w('left').computeSize=()=>[1,1];n.onConfigure();assert.equal(w('left').type,'converted-widget');assert.equal(w('left').computeSize()[0],1);
n.inputs=[{name:'task',link:1}];n.onConnectionsChange();assert.notEqual(w('right').hidden,true);
class Inputs extends Node {constructor(){super();this.comfyClass='SwwanQwen21PrunaInputs';this.widgets=['image_filename','mask_filename','protect_filename','reference_filenames'].map(name=>({name,value:'kept',type:'text'}));}}
extensions[0].beforeRegisterNodeDef(Inputs,{name:'SwwanQwen21PrunaInputs'});
const i=new Inputs();i.inputs=[{name:'config',link:5}];i.graph={links:{5:{origin_id:600}},getNodeById:()=>n};n.inputs=[];set('task','photoreal');i.onConfigure();assert(i.widgets.every(w=>w.hidden));assert(i.widgets.every(w=>w.value==='kept'));
set('task','remove');i.onConfigure();assert.notEqual(i.widgets.find(w=>w.name==='mask_filename').hidden,true);
console.log('PASS: eight-task visibility, active LoRA fields, text inputs, converted ports and retained hidden values');
