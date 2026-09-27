import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
const extensions=[];let queued;
const app={registerExtension:extension=>extensions.push(extension),graph:{getNodeById:()=>seedNode}};
const api={async queuePrompt(number,prompt){queued=prompt;return {prompt_id:'test'};}};
const context=vm.createContext({app,api,structuredClone,console,Math});
for(const file of ['swwan_seed','swwan_dynamic_inputs']){
    const source=fs.readFileSync(new URL(`../web/js/${file}.js`,import.meta.url),'utf8').replace(/^import .*;\n/gm,'').replace(/^export /gm,'');
    vm.runInContext(source,context);
}
assert.equal(vm.runInContext('resolveSeed(-2, 8)',context),9);
assert.equal(vm.runInContext('resolveSeed(-3, 8)',context),7);
assert.equal(vm.runInContext('resolveSeed(-3, 0)',context),1125899906842624);
assert.equal(vm.runInContext('resolveSeed(-1, null, () => 42)',context),42);
const seedNode={properties:{swwan_last_seed:8}};
extensions.find(x=>x.name==='Swwan.Seed').setup();
const original={output:{'7':{class_type:'SwwanSeed',inputs:{seed:-2}},'8':{class_type:'Seed (rgthree)',inputs:{seed:-2}}},workflow:{nodes:[{id:7,widgets_values:[-2],widgets_values_named:{seed:-2}}]}};
await api.queuePrompt(0,original);
assert.equal(original.output['7'].inputs.seed,-2);assert.equal(queued.output['7'].inputs.seed,9);
assert.equal(queued.output['8'].inputs.seed,-2);assert.equal(queued.workflow.nodes[0].widgets_values[0],9);
assert.equal(queued.workflow.nodes[0].widgets_values_named.seed,9);
assert.equal(seedNode.properties.swwan_last_seed,9);
const saved=JSON.parse(JSON.stringify(seedNode));assert.equal(saved.properties.swwan_last_seed,9);
const node={widgets:[{name:'inputcount',value:4}],inputs:[{name:'image_1',type:'IMAGE',link:1},{name:'image_2',type:'IMAGE',link:null}],
 addInput(name,type){this.inputs.push({name,type,link:null});},removeInput(index){this.inputs.splice(index,1);},setDirtyCanvas(){}};
context.testNode=node;vm.runInContext('updateInputs(testNode)',context);assert.equal(node.inputs.length,4);
node.inputs[3].link=8;node.widgets[0].value=2;vm.runInContext('updateInputs(testNode)',context);
assert.equal(node.inputs.length,4);assert.equal(node.widgets[0].value,4);assert.equal(node.inputs[3].link,8);
node.inputs[3].link=null;node.widgets[0].value=2;vm.runInContext('updateInputs(testNode)',context);assert.equal(node.inputs.length,2);
class SeedNode {
    constructor(){this.widgets=[{name:'seed',value:0},{name:'control_after_generate',value:'randomize'}];this.properties={};}
    addWidget(type,name,value,callback,options){this.widgets.push({type,name,value,callback,options});}
}
extensions.find(x=>x.name==='Swwan.Seed').beforeRegisterNodeDef(SeedNode,{name:'SwwanSeed'});
const nativeSeed=new SeedNode();nativeSeed.onNodeCreated();
assert.equal(nativeSeed.widgets.some(w=>w.name==='control_after_generate'),false);
nativeSeed.widgets.find(w=>w.name==='Increment each time').callback();assert.equal(nativeSeed.widgets[0].value,-2);
console.log('PASS: seed modes/metadata/ownership and dynamic connected input preservation');
