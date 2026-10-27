import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { COLORS, LAYERS, RAYS, makeLayers, makeAtom, makeGravity, makeSpeeds, disposeGroup } from './model.js';

const $ = selector => document.querySelector(selector);
const viewport = $('#viewport'), labelHost = $('#labels');
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0x000000, 0);
viewport.prepend(renderer.domElement);
renderer.domElement.setAttribute('aria-label', 'Интерактивная 3D-сцена: вращение мышью, масштаб колесом');
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(43, 1, .1, 150);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = .07;
controls.minDistance = 6;
controls.maxDistance = 42;
controls.autoRotateSpeed = .4;
const state = { scene: 'layers', time: 0, playing: true, speed: 1, cut: true, explode: 0, layer: 0, assembly: 0, assembling: false, gravity: 'both', race: 0 };
let model, labelElements = [], lastTime = performance.now(), lastUi = 0, viewFit = 1;
const sceneData = {
  layers: { kicker: '01 / АРХИТЕКТУРА ПРОСТРАНСТВА', title: 'Вложенные миры', caption: 'Шесть эфирных слоёв между двумя кластерами', bottom: 'Выберите слой в списке или нажмите на оболочку', factory: () => makeLayers({cut:state.cut}) },
  atom: { kicker: '02 / ФОРМИРОВАНИЕ СТРУКТУРЫ', title: 'От поля к атому', caption: 'Эфирные поля → квантовое поле → условная структура атома', bottom: 'Запустите сборку или изменяйте её этап вручную', factory: makeAtom },
  gravity: { kicker: '03 / ДВА ТИПА ВЗАИМОДЕЙСТВИЯ', title: 'Внутри и вокруг', caption: 'Направления и траектории — иллюстрация описаний из текста', bottom: 'Переключайте поля А и Б для сравнения', factory: makeGravity },
  speeds: { kicker: '04 / СКОРОСТИ ИЗ ИСХОДНЫХ ТЕКСТОВ', title: 'Общий старт. Разный темп.', caption: 'Одна дистанция · линейная шкала скорости · общий секундомер', bottom: '«Сначала» повторяет запуск всех пяти импульсов', factory: makeSpeeds },
};
function metrics(items) { return items.map(([label,value])=>`<div class="metric"><span>${label}</span><b>${value}</b></div>`).join(''); }
function info(tag,title,text,items,source) {
  $('#info-tag').textContent=tag;$('#info-title').textContent=title;$('#info-text').textContent=text;$('#info-metrics').innerHTML=metrics(items);$('#info-source').textContent=source;
}
function selectLayer(index) {
  state.layer=index;
  document.querySelectorAll('[data-layer]').forEach(b=>{const selected=Number(b.dataset.layer)===index;b.classList.toggle('selected',selected);b.setAttribute('aria-pressed',selected);});
  const l=LAYERS[index];info(l.tag,l.name,l.text,[l.metric,['Энергия океана по тексту','700×']],'Источник: «Текстовый документ (2).txt». Значение 700× относится ко всей прослойке.');
}
function buildControls() {
  const host=$('#scene-controls');
  if(state.scene==='layers') {
    host.innerHTML=`<label class="toggle"><input id="cut" type="checkbox" ${state.cut?'checked':''}>Разрез оболочек</label><label class="range-label" for="explode">Раздвинуть слои <output id="explode-value">${Math.round(state.explode*100)}%</output></label><input id="explode" type="range" min="0" max="1" step="0.01" value="${state.explode}"><p class="section-label">ЭФИРНАЯ ПРОСЛОЙКА</p><div class="layer-list">${LAYERS.map((l,i)=>`<button class="layer-button" data-layer="${i}" aria-pressed="false"><span class="swatch" style="--swatch:${COLORS[i]}"></span>${l.short}<small>0${i+1}</small></button>`).join('')}</div>`;
    $('#cut').addEventListener('change',e=>{state.cut=e.target.checked;rebuildModel();});
    $('#explode').addEventListener('input',e=>{
      const previous=state.explode;state.explode=Number(e.target.value);
      camera.position.sub(controls.target).multiplyScalar((1+.38*state.explode)/(1+.38*previous));
      controls.target.y+=(state.explode-previous)*2.1;camera.position.add(controls.target);controls.update();
      $('#explode-value').textContent=Math.round(state.explode*100)+'%';
    });
    host.querySelectorAll('[data-layer]').forEach(b=>b.addEventListener('click',()=>selectLayer(Number(b.dataset.layer))));selectLayer(state.layer);
  } else if(state.scene==='atom') {
    host.innerHTML=`<p class="section-label">СОСТАВ ОБЩЕГО ПОЛЯ ПО ТЕКСТУ</p><div class="composition"><span></span><span></span></div><div class="composition-legend"><span>73% · эфирные поля</span><span>27% · атомы</span></div><label class="range-label" for="assembly">Этап сборки <output id="assembly-value">${Math.round(state.assembly*100)}%</output></label><input id="assembly" type="range" min="0" max="1" step=".01" value="${state.assembly}"><button id="assemble" class="wide-button">Запустить сборку →</button>`;
    $('#assembly').addEventListener('input',e=>{state.assembly=Number(e.target.value);state.assembling=false;syncDynamicUi();});
    $('#assemble').addEventListener('click',()=>{state.assembly=0;state.assembling=true;setPlaying(true);});
    info('Схема преобразования','Поле становится структурой','Частицы двух цветов сходятся в центре. Текст связывает образование атомов с последовательным объединением квантов. Здесь показан только этот принцип: состав и орбиты атома условны.',[['Эфирные поля','73%'],['Излучение атомов','27%']],'Источники: «Текстовый документ (3).txt», «Текстовый документ (4).txt».');
  } else if(state.scene==='gravity') {
    host.innerHTML=`<p class="section-label">ПОКАЗАТЬ ПОЛЯ</p><div class="choice-row">${[['both','А + Б'],['a','Только А'],['b','Только Б']].map(([value,label])=>`<button data-gravity="${value}" class="${state.gravity===value?'selected':''}" aria-pressed="${state.gravity===value}">${label}</button>`).join('')}</div>`;
    host.querySelectorAll('[data-gravity]').forEach(b=>b.addEventListener('click',()=>{state.gravity=b.dataset.gravity;host.querySelectorAll('[data-gravity]').forEach(x=>{const on=x===b;x.classList.toggle('selected',on);x.setAttribute('aria-pressed',on);});}));
    info('Авторская классификация','Два типа гравитонов','А — направленные внутрь спирали, которым приписывается связывание атомов. Б — поле вокруг составного объекта. Движущиеся точки показывают направление потока; силы и орбиты здесь не рассчитываются.',[['А · роль','Связи'],['Б · роль','Внешнее поле']],'Источник: «Текстовый документ (6).txt».');
  } else {
    host.innerHTML=`<div class="speeds-list">${RAYS.map(r=>`<div class="speed-row"><span><i class="swatch" style="--swatch:${r.color}"></i>${r.name}</span><b>${String(r.speed).replace('.',',')} c</b></div>`).join('')}</div><div class="progress-track"><i id="race-progress"></i></div><div class="composition-legend"><span>Общий отсчёт</span><span id="race-time">0,00 τ</span></div><button id="race-restart" class="wide-button">Повторить запуск →</button>`;
    $('#race-restart').addEventListener('click',()=>{state.race=0;setPlaying(true);});
    info('Сравнение, не измерение','Скорости из текстов','Все импульсы стартуют одновременно. Свет проходит всю дистанцию за 1 τ; демонстрация растягивает этот интервал до 12 секунд при темпе 1×. Быстрые импульсы остаются на финише.',[['Единица скорости','c'],['Свет проходит путь','1 τ']],'Источники: «Текстовый документ.txt», «Текстовый документ (2).txt». Значения сверх c относятся к гипотезе авторов.');
  }
}
function rebuildModel() {
  if(model) disposeGroup(model.group);
  model=sceneData[state.scene].factory();scene.add(model.group);
  labelHost.replaceChildren();
  labelElements=model.labels.map(label=>{const el=document.createElement('div');el.className='world-label';el.textContent=label.text;el.style.setProperty('--label-color',label.color);labelHost.append(el);return el;});
  model.update(state.time,state);
}
function resetCamera() {
  camera.position.set(...model.camera).multiplyScalar(viewFit);controls.target.set(...model.target);
  if(state.scene==='layers'){camera.position.multiplyScalar(1+.38*state.explode);camera.position.y+=state.explode*2.1;controls.target.y+=state.explode*2.1;}
  controls.update();
}
function setScene(id) {
  state.scene=id;state.time=0;state.race=0;state.assembly=0;state.assembling=false;
  const data=sceneData[id];$('#scene-kicker').textContent=data.kicker;$('#scene-title').textContent=data.title;$('#scene-caption').textContent=data.caption;$('#bottom-note').textContent=data.bottom;
  document.querySelectorAll('[data-scene]').forEach(b=>{const selected=b.dataset.scene===id;b.classList.toggle('active',selected);b.setAttribute('aria-pressed',selected);});
  rebuildModel();buildControls();resize();resetCamera();
}
function setPlaying(value) {
  state.playing=value;$('#pause').textContent=value?'Ⅱ':'▶';$('#pause').setAttribute('aria-label',value?'Приостановить анимацию':'Продолжить анимацию');$('#status').textContent=value?'В ДВИЖЕНИИ':'ПАУЗА';
}
function syncDynamicUi() {
  if(state.scene==='atom') {$('#assembly').value=state.assembly;$('#assembly-value').textContent=Math.round(state.assembly*100)+'%';$('#assemble').textContent=state.assembling?'Сборка…':state.assembly>=1?'Повторить сборку ↺':'Запустить сборку →';}
  if(state.scene==='speeds') {$('#race-progress').style.width=Math.min(100,state.race*100*.85)+'%';$('#race-time').textContent=state.race.toFixed(2).replace('.',',')+' τ';}
}
function resize() {
  const {width,height}=viewport.getBoundingClientRect();renderer.setSize(width,height);camera.aspect=width/height;camera.updateProjectionMatrix();
  const horizontalFit=state.scene==='speeds'?1.9:state.scene==='gravity'?1.8:1;
  const nextFit=Math.max(1,horizontalFit/camera.aspect);
  camera.position.sub(controls.target).multiplyScalar(nextFit/viewFit).add(controls.target);viewFit=nextFit;
  controls.maxDistance=Math.max(42,32*viewFit);controls.update();
}
new ResizeObserver(resize).observe(viewport);
document.querySelectorAll('[data-scene]').forEach(b=>b.addEventListener('click',()=>setScene(b.dataset.scene)));
$('#pause').addEventListener('click',()=>setPlaying(!state.playing));
$('#speed').addEventListener('input',e=>{state.speed=Number(e.target.value);$('#speed-value').textContent=String(state.speed).replace('.',',')+'×';});
$('#rotate').addEventListener('change',e=>{controls.autoRotate=e.target.checked;});
$('#reset').addEventListener('click',resetCamera);
$('#restart').addEventListener('click',()=>{state.time=0;state.race=0;state.assembly=0;state.assembling=false;syncDynamicUi();});
$('#fullscreen').addEventListener('click',async()=>{
  try { if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen(); }
  catch {$('#status').textContent='ПОЛНЫЙ ЭКРАН НЕДОСТУПЕН';}
});
$('#capture').addEventListener('click',()=>{
  renderer.render(scene,camera);
  const canvas=document.createElement('canvas');canvas.width=renderer.domElement.width;canvas.height=renderer.domElement.height;
  const ctx=canvas.getContext('2d');ctx.fillStyle='#0b121d';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.drawImage(renderer.domElement,0,0);
  const scale=canvas.width/viewport.clientWidth;
  ctx.fillStyle='#e4edf5';ctx.font=`${20*scale}px Segoe UI`;ctx.fillText(sceneData[state.scene].title,24*scale,35*scale);
  ctx.fillStyle='#96aabc';ctx.font=`${10*scale}px Segoe UI`;ctx.fillText('Концептуальная модель · геометрия и масштаб условны',24*scale,55*scale);
  labelElements.forEach(el=>{if(el.hidden)return;const x=parseFloat(el.style.left)*scale,y=parseFloat(el.style.top)*scale;ctx.font=`${10*scale}px Segoe UI`;ctx.textAlign='center';ctx.fillText(el.textContent,x,y);});
  const a=document.createElement('a');a.download=`field-atlas-${state.scene}.png`;a.href=canvas.toDataURL('image/png');a.click();
});
const raycaster=new THREE.Raycaster(),pointer=new THREE.Vector2();let pointerStart;
renderer.domElement.addEventListener('pointerdown',e=>{pointerStart=[e.clientX,e.clientY];});
renderer.domElement.addEventListener('pointerup',e=>{
  if(state.scene!=='layers'||!pointerStart||Math.hypot(e.clientX-pointerStart[0],e.clientY-pointerStart[1])>5)return;
  const rect=renderer.domElement.getBoundingClientRect();pointer.set((e.clientX-rect.left)/rect.width*2-1,-(e.clientY-rect.top)/rect.height*2+1);raycaster.setFromCamera(pointer,camera);
  const hits=raycaster.intersectObjects(model.picks);if(hits.length)selectLayer(hits[0].object.userData.layer);
});
renderer.domElement.addEventListener('webglcontextlost',e=>{e.preventDefault();setPlaying(false);$('#failure').hidden=false;$('#failure').textContent='Графический контекст потерян. Обновите страницу, чтобы восстановить 3D.';});
setScene('layers');resize();setPlaying(true);
const projected=new THREE.Vector3();
function frame(now) {
  const dt=Math.min((now-lastTime)/1000,.05);lastTime=now;
  if(state.playing&&!document.hidden) {
    state.time+=dt*state.speed;
    if(state.assembling) {state.assembly=Math.min(1,state.assembly+dt*state.speed/8);if(state.assembly===1)state.assembling=false;}
    if(state.scene==='speeds')state.race=Math.min(1/.85,state.race+dt*state.speed/12);
  }
  model.update(state.time,state);
  const auto=controls.autoRotate;controls.autoRotate=auto&&state.playing;controls.update(dt);controls.autoRotate=auto;
  renderer.render(scene,camera);
  const width=viewport.clientWidth,height=viewport.clientHeight;
  model.labels.forEach((label,i)=>{projected.copy(label.position).project(camera);const x=(projected.x*.5+.5)*width,y=(-projected.y*.5+.5)*height;const el=labelElements[i];el.hidden=projected.z>1||projected.z< -1||x<0||x>width||y<15||y>height-10;el.style.left=x+'px';el.style.top=y+'px';});
  if(now-lastUi>100){syncDynamicUi();lastUi=now;}
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
// Read-only diagnostics for local browser verification.
window.fieldAtlas = { snapshot:()=>({ ...state, revision:THREE.REVISION, objects:model.group.children.length, drawCalls:renderer.info.render.calls, triangles:renderer.info.render.triangles, camera:camera.position.toArray(), canvas:[renderer.domElement.width,renderer.domElement.height] }) };
