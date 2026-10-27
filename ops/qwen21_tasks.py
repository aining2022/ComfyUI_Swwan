# SPDX-License-Identifier: MIT
"""Shared, inspectable task contracts. No models or GPU libraries imported here."""
import math

TASKS = ('remove','watermark','edit','local_edit','enhance','cutout','face_swap','body_swap','outfit_swap','outpaint','angles')
LORAS = {
 'remove':'Qwen2.1_万物移除Remove_GH.safetensors',
 'watermark':'Qwen21-水印移除watermark_remover_v2.safetensors',
 'consistency':'Qwen2.1 一致性无偏移_GH_V1.0.safetensors',
 'enhance':'unblur-background-qwen-image-2.1-edit-lora.safetensors',
 'cutout':'心宝强化抠图.safetensors',
 'face_v1.0':'bfs_head_v1_qwen_2.1.safetensors',
 'face_v1.1':'bfs_head_v1.1_qwen_2.1.safetensors',
 'body_swap':'qwen21-unified/bfs_body_swap_v1.0_qwen_2.1.safetensors',
 'outfit_swap':'qwen21-unified/qwen-image-2.1-outfit-swap.safetensors',
 'outpaint':'qwen-image-2.1-outpaint-v2.safetensors',
 'angles':'qwen21-unified/angles_v2_full_qwen21_multiple_angles_v2_qwen21_multiple_angles_v2_000001500.safetensors',
}
SIGMA8 = (1., .978453, .95418, .926626, .89508, .845148, .704534, .414568)
SIGMA9 = SIGMA8 + (.15,)
PRESETS = ('turbo9','official8','experimental12','experimental16')
AZIMUTHS = ('front view','front-right view','front-right quarter view','right side view',
 'back-right quarter view','back-right view','back view','back-left view',
 'back-left quarter view','left side view','front-left quarter view','front-left view')
ELEVATIONS = {0:'eye-level shot',30:'elevated shot',60:'high-angle shot',90:'top-down shot'}
BASE_PROMPTS = {
 'remove':'移除图像中的绿色区域',
 'watermark':'remove watermark',
 'enhance':'bring the background into sharp focus and full detail',
 'cutout':'Xinbao,抠出[红色定位轮廓所选定的主体]，移除背景和所有外部遮挡物，移除红色定位轮廓标记，最终仅保留所选主体及其自身组成部分，补全其被遮挡的部分并显示完整主体；保持原有位置、比例、视角、颜色、材质和未遮挡细节不变，输出透明背景PNG。',
 'body_swap':'body_swap: start with <image1> as the base image, keeping its lighting, environment, and background. replace the body from <image1> with the body from <image2>, strictly preserving the clothing, body shape and proportions from <image2>. copy the pose, direction of the eye, head rotation, micro expressions from <image1>',
 'outfit_swap':'Dress the person in image 1 in the outfit from image 2. Preserve the identity, pose, body proportions, background and lighting of image 1.',
 'outpaint':'Outpaint the image: replace the solid gray areas with a seamless continuation of the scene, keeping the existing picture unchanged.',
}


def sigmas(preset):
    if preset not in PRESETS: raise ValueError('Unknown sampling preset')
    if preset=='official8': return list(SIGMA8)+( [0.] )
    if preset=='turbo9': return list(SIGMA9)+[0.]
    n=int(preset.removeprefix('experimental'))
    values=[]
    for i in range(n):
        t=i*(len(SIGMA9)-1)/(n-1); lo=int(t); hi=min(lo+1,len(SIGMA9)-1)
        values.append(math.exp(math.log(SIGMA9[lo])*(1-(t-lo))+math.log(SIGMA9[hi])*(t-lo)))
    return values+[0.]


def face_prompt(version):
    expression='1' if version=='v1.0' else '2'
    return ('head_swap: start with <image1> as the base image, keeping its lighting, environment, and background. '
            'remove the head from <image1> completely and replace it with the head from <image2>, strictly preserving '
            'the hair, eye color, nose structure from <image2>. copy the direction of the eye and head rotation from '
            f'<image1>; preserve the micro expressions from <image{expression}>, high quality, sharp details, 4k')


def resolve(task='remove', extra_prompt='', lora_strength=1., consistency='auto', seed=42,
            resolution=1024, sampling_preset='turbo9', people=1, swap_head=False, forehead=False,
            bfs_version='v1.1', skin_match=False, watermark_mode='local', left=128, right=128,
            top=128, bottom=128, azimuth=0, elevation=0, close_up=False, **kwargs):
    if task not in TASKS: raise ValueError('Unknown task: '+task)
    if not math.isfinite(float(lora_strength)) or not 0<=lora_strength<=2: raise ValueError('lora_strength must be in 0..2')
    if resolution<512 or resolution>2048 or resolution%32: raise ValueError('resolution must be a multiple of 32 in 512..2048')
    if consistency not in ('auto','on','off',True,False): raise ValueError('consistency must be auto, on or off')
    enabled=(task=='local_edit') if consistency=='auto' else consistency in ('on',True)
    enabled=enabled and task in ('edit','local_edit')
    if task in ('edit','local_edit') and not extra_prompt.strip(): raise ValueError('请补充具体编辑指令 extra_prompt，尚未开始采样。')
    if not 1<=people<=6: raise ValueError('people must be in 1..6')
    if bfs_version not in ('v1.0','v1.1'): raise ValueError('bfs_version must be v1.0 or v1.1')
    if watermark_mode not in ('local','full'): raise ValueError('watermark_mode must be local or full')
    if azimuth not in range(0,360,30) or elevation not in ELEVATIONS: raise ValueError('Unsupported camera angle')
    if any(not 0<=v<=4096 for v in (left,right,top,bottom)): raise ValueError('Outpaint margins must be in 0..4096')
    if task=='outpaint' and not any((left,right,top,bottom)): raise ValueError('At least one outpaint margin must be positive')
    base=BASE_PROMPTS.get(task,'')
    if task=='face_swap': base=face_prompt(bfs_version)
    if task=='angles': base=f'<mva> {AZIMUTHS[azimuth//30]}, {ELEVATIONS[elevation]}' + (' close-up' if close_up else '')
    prompt=base+ ('\n'+extra_prompt if base and extra_prompt else extra_prompt)
    key=('face_'+bfs_version) if task=='face_swap' else ('consistency' if enabled else task)
    lora=LORAS.get(key) if task not in ('edit','local_edit') or enabled else None
    return dict(task=task,extra_prompt=extra_prompt,lora_strength=float(lora_strength),consistency=enabled,
        seed=int(seed),resolution=int(resolution),sampling_preset=sampling_preset,sigmas=sigmas(sampling_preset),
        people=int(people),swap_head=bool(swap_head),forehead=bool(forehead),bfs_version=bfs_version,
        skin_match=bool(skin_match),watermark_mode=watermark_mode,left=int(left),right=int(right),top=int(top),bottom=int(bottom),
        azimuth=azimuth,elevation=elevation,close_up=bool(close_up),prompt=prompt,lora=lora if lora_strength else None,
        configured_lora=lora,encoder_resolution=0)
