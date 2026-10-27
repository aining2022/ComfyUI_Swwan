import { app } from '../../scripts/app.js';
import { api } from '../../scripts/api.js';

function update(node) {
    const task = node.widgets?.find(w => w.name === 'task')?.value;
    const visible = new Set(['task','extra_prompt','lora_strength','seed','resolution','sampling_preset']);
    if (task === 'edit' || task === 'local_edit') visible.add('consistency');
    if (task === 'face_swap') ['people','swap_head','forehead','bfs_version','skin_match'].forEach(k => visible.add(k));
    if (task === 'watermark') visible.add('watermark_mode');
    if (task === 'outpaint') ['left','right','top','bottom'].forEach(k => visible.add(k));
    if (task === 'angles') ['azimuth','elevation','close_up'].forEach(k => visible.add(k));
    for (const w of node.widgets || []) {
        if (w.type?.startsWith('converted-widget')) continue;
        if (!w.qwen21Original) w.qwen21Original = { type:w.type, computeSize:w.computeSize, hidden:w.hidden };
        w.hidden = visible.has(w.name) ? w.qwen21Original.hidden : true;
        w.type = visible.has(w.name) ? w.qwen21Original.type : 'swwan-hidden';
        w.computeSize = visible.has(w.name) ? w.qwen21Original.computeSize : () => [0,-4];
    }
    node.setSize([Math.max(420,node.size[0]),node.computeSize()[1]]);
    node.setDirtyCanvas(true,true);
}
app.registerExtension({
    name:'Swwan.Qwen21Controls',
    async beforeRegisterNodeDef(type, data) {
        if (data.name === 'SwwanQwen21Controls') {
            const created=type.prototype.onNodeCreated;
            type.prototype.onNodeCreated=function() {
                const result=created?.apply(this,arguments);
                const task=this.widgets?.find(w=>w.name==='task');
                if (task) { const callback=task.callback; task.callback=(...args)=>{callback?.apply(task,args);update(this);}; }
                this.swwanRefreshModes=()=>update(this);
                update(this); return result;
            };
            const configured=type.prototype.onConfigure;
            type.prototype.onConfigure=function() { const r=configured?.apply(this,arguments);update(this);return r; };
        }
        if (data.name === 'SwwanQwen21References') {
            const created=type.prototype.onNodeCreated;
            type.prototype.onNodeCreated=function() {
                const result=created?.apply(this,arguments);
                this.addWidget('button','上传参考图（按选择顺序，最多 6 张）',null,()=>{
                    const picker=document.createElement('input');picker.type='file';picker.accept='image/*';picker.multiple=true;
                    picker.onchange=async()=>{
                        const files=Array.from(picker.files||[]);
                        if (files.length>6) {alert('最多 6 张参考图');return;}
                        try {
                            const names=[];
                            for (const file of files) {
                                const body=new FormData();body.append('image',file);body.append('subfolder','qwen21-references');
                                const response=await api.fetchApi('/upload/image',{method:'POST',body});
                                if (!response.ok) throw new Error(await response.text());
                                const data=await response.json();names.push([data.subfolder,data.name].filter(Boolean).join('/'));
                            }
                            this.widgets.find(w=>w.name==='filenames').value=names.join('\n');this.setDirtyCanvas(true);
                        } catch(e) {alert(String(e));}
                    };
                    picker.click();
                },{serialize:false});
                return result;
            };
        }
    }
});
