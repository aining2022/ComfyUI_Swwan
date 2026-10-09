import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
const extensions=[];let queued,queueCall,queueError;
const app={registerExtension:extension=>extensions.push(extension),graph:{getNodeById:()=>seedNode}};
const api={async queuePrompt(number,prompt,...args){queued=prompt;queueCall={receiver:this,number,args};if(queueError)throw queueError;return {prompt_id:'test'};}};
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

// Reactive frontend objects serialize to JSON, but structuredClone rejects them.
function reactive(value,cache=new WeakMap()){
    if(!value || typeof value!=='object')return value;
    if(!cache.has(value))cache.set(value,new Proxy(value,{get(target,key,receiver){return reactive(Reflect.get(target,key,receiver),cache);}}));
    return cache.get(value);
}
const noSeed=reactive({output:{'1':{class_type:'KSampler',inputs:{seed:42}}},workflow:{nodes:[{id:1,properties:{label:'sampler'}}]}});
assert.throws(()=>structuredClone(noSeed),{name:'DataCloneError'});
const noSeedBefore=JSON.stringify(noSeed);
const options={partialExecutionTargets:['1']};
const receiver={};
assert.equal((await api.queuePrompt.call(receiver,-1,noSeed,options)).prompt_id,'test');
assert.equal(queued,noSeed,'Workflows without SwwanSeed must pass through unchanged');
assert.equal(JSON.stringify(queued),noSeedBefore);
assert.equal(queueCall.receiver,receiver);assert.equal(queueCall.number,-1);assert.equal(queueCall.args[0],options);
assert.equal(seedNode.properties.swwan_last_seed,9);

const linked=reactive({output:{'7':{class_type:'SwwanSeed',inputs:{seed:['1',0]}}},workflow:{nodes:[{id:7,widgets_values:[-2]}]}});
await api.queuePrompt(0,linked);
assert.equal(queued,linked,'Linked seeds must remain under upstream control');
assert.equal(seedNode.properties.swwan_last_seed,9);

const proxied=reactive({
    output:{'7':{class_type:'SwwanSeed',inputs:{seed:-2},_meta:{title:'Seed'}},'8':{class_type:'Seed (rgthree)',inputs:{seed:-2}}},
    workflow:{nodes:[{id:7,widgets_values:[-2,'preserved'],widgets_values_named:{seed:-2,other:'preserved'},properties:{swwan_last_seed:9,other:'preserved'}},{id:8,widgets_values:[-2]}],extra:{renderer:'preserved'}},
    extra:{extension:'preserved'},
});
const proxiedBefore=JSON.stringify(proxied);
await api.queuePrompt(0,proxied,options);
assert.notEqual(queued,proxied);assert.notEqual(queued.output,proxied.output);
assert.notEqual(queued.output['7'],proxied.output['7']);assert.notEqual(queued.output['7'].inputs,proxied.output['7'].inputs);
assert.equal(queued.output['7'].inputs.seed,10);assert.equal(queued.output['7']._meta,proxied.output['7']._meta);
assert.equal(queued.workflow.nodes[0].widgets_values[0],10);assert.equal(queued.workflow.nodes[0].widgets_values[1],'preserved');
assert.equal(queued.workflow.nodes[0].widgets_values_named.seed,10);assert.equal(queued.workflow.nodes[0].widgets_values_named.other,'preserved');
assert.equal(queued.workflow.nodes[0].properties.swwan_last_seed,10);assert.equal(queued.workflow.nodes[0].properties.other,'preserved');
assert.equal(queued.output['8'],proxied.output['8']);assert.equal(queued.workflow.nodes[1],proxied.workflow.nodes[1]);
assert.equal(queued.workflow.extra,proxied.workflow.extra);assert.equal(queued.extra,proxied.extra);
assert.equal(JSON.stringify(proxied),proxiedBefore,'Seed resolution must not mutate the submitted source');
assert.doesNotThrow(()=>JSON.stringify(queued));assert.equal(seedNode.properties.swwan_last_seed,10);
assert.equal(queueCall.args[0],options);

// Failed submission must neither consume an increment nor replace the API error.
queueError=new Error('queue rejected');
await assert.rejects(api.queuePrompt(0,proxied),error=>error===queueError);
queueError=undefined;
assert.equal(seedNode.properties.swwan_last_seed,10);assert.equal(JSON.stringify(proxied),proxiedBefore);
await api.queuePrompt(0,proxied);
assert.equal(queued.output['7'].inputs.seed,11);assert.equal(seedNode.properties.swwan_last_seed,11);

// Ordinary prompts can also carry nested proxies or non-serializable runtime helpers.
const mixed={output:{'7':{class_type:'SwwanSeed',inputs:{seed:123}}},workflow:{nodes:[{id:7,properties:reactive({label:'fixed'})}]},helper(){}};
await api.queuePrompt(0,mixed);
assert.equal(queued.output['7'].inputs.seed,123);assert.equal(queued.helper,mixed.helper);
assert.equal(queued.workflow.nodes[0].properties.swwan_last_seed,123);
assert.equal(mixed.workflow.nodes[0].properties.swwan_last_seed,undefined);
assert.equal(seedNode.properties.swwan_last_seed,123);
const outputOnly={output:{'7':{class_type:'SwwanSeed',inputs:{seed:-3}}}};
await api.queuePrompt(0,outputOnly);
assert.equal(queued.output['7'].inputs.seed,122);assert.equal(queued.workflow,undefined);
assert.equal(outputOnly.output['7'].inputs.seed,-3);
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
console.log('PASS: seed modes/metadata/ownership, reactive prompt queuing, failure state and dynamic connected input preservation');
