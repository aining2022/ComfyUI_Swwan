import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
const extensions=[];
const graph={nodes:new Map(),getNodeById(id){return this.nodes.get(id);}};
const app={graph,registerExtension:e=>extensions.push(e),async graphToPrompt(){return {output:{1:{class_type:'SwwanImageMatte',inputs:{}}},workflow:{nodes:[]}};}};
vm.runInNewContext(readFileSync(new URL('../web/js/swwan_widget_values.js',import.meta.url),'utf8').replace(/^import .*;\n/gm,''),{app});
const extension=extensions[0];
const catalog=JSON.parse(readFileSync(new URL('../docs/node-catalog.json',import.meta.url),'utf8'));
function node(id, widgets){
 const data=catalog.find(n=>n.id===id);
 class Node {constructor(){this.id=1;this.widgets=widgets;this.inputs=[];} onSerialize(info){info.native=true;} }
 extension.beforeRegisterNodeDef(Node,{name:id,category:data.category,input:data.schema});
 return new Node();
}
for(const present of [true,false]){
 const n=node('SwwanImageMatte',present?[{name:'background_color',value:'#364254'}]:[]);
 n.onConfigure({widgets_values_named:{background_color:'#ffffff',crop_mask:false,crop_factor:1.2,stroke_width:0,fill_holes:true}});
 graph.nodes.set(1,n);
 const info={widgets_values:['native']};n.onSerialize(info);
 assert.equal(info.widgets_values_named.background_color,'#ffffff');assert.deepEqual(info.widgets_values,['native']);assert.equal(info.native,true);
 const reloaded=node('SwwanImageMatte',[]);reloaded.onConfigure(info);const again={};reloaded.onSerialize(again);
 assert.deepEqual(again.widgets_values_named,info.widgets_values_named);
}
extension.setup();
assert.equal((await app.graphToPrompt()).output[1].inputs.background_color,'#ffffff');
graph.nodes.get(1).inputs=[{name:'background_color',link:5}];
assert.equal((await app.graphToPrompt()).output[1].inputs.background_color,undefined);
const converted=node('SwwanDrawMaskOnImage',[{name:'opacity',value:.4,type:'converted-widget',serialize:false}]);
converted.onConfigure({widgets_values_named:{color:'255, 255, 255',device:'cpu',opacity:.7}});
const saved={};converted.onSerialize(saved);assert.equal(saved.widgets_values_named.opacity,.7);
assert.throws(()=>converted.onConfigure({widgets_values_named:{opacity:'cpu'}}),/invalid opacity/);
assert.throws(()=>converted.onConfigure({widgets_values_named:{device:'cuda'}}),/invalid device/);
assert.throws(()=>converted.onConfigure({widgets_values_named:{unknown:1}}),/unknown saved field/);
assert.throws(()=>converted.onConfigure({widgets_values_named:{opacity:true}}),/invalid opacity/);
class Other {} extension.beforeRegisterNodeDef(Other,{name:'Other',category:'Other/Image'});assert.equal(Other.prototype.onSerialize,undefined);
console.log('PASS: named save/reload, missing COLORCODE fallback, connected colors, converted widgets and strict validation');
