// Check real import URLs rather than removing imports as VM behavior tests do.
import assert from 'node:assert/strict';
import {readFileSync,readdirSync} from 'node:fs';
const directory=new URL('../web/js/',import.meta.url);
let imports=0;
for(const file of readdirSync(directory).filter(name=>name.endsWith('.js'))){
 const source=readFileSync(new URL(file,directory),'utf8');
 for(const [,specifier] of source.matchAll(/^import .*? from ['"]([^'"]+)['"];$/gm)){
  imports++;
  for(const prefix of ['/','/comfyui/']){
   const module=new URL(`${prefix}extensions/ComfyUI_Swwan/${file}`,'http://test.invalid');
   const target=new URL(specifier,module);
   assert.equal(target.pathname,`${prefix}scripts/${specifier.split('/').at(-1)}`,`${file}: import must retain ${prefix}`);
  }
 }
}
assert.ok(imports>=7,'All frontend entry points and Seed API import must be checked');
console.log(`PASS: ${imports} frontend imports retain root and /comfyui/ base paths`);
