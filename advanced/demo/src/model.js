import * as THREE from 'three';

export const COLORS = ['#73e8cf', '#63bedc', '#709be8', '#9b92ec', '#bd8bd9', '#dd9bc7'];
export const LAYERS = [
  { name: 'Торсионная оболочка', short: 'Торсионные поля', tag: 'Слой 1 · у физического мира', text: 'Внешняя оболочка эфирной прослойки со стороны физического мира. По тексту, через неё происходит формирование физических квантовых полей.', metric: ['Спираль в описании', '12,5 витка'] },
  { name: 'Лептонные поля', short: 'Лептонные поля', tag: 'Слой 2 · прототип полей', text: 'Следуют за торсионными полями и описываются как прототип физических квантовых полей. Их размещение вторым слоем — интерпретация последовательности текста.', metric: ['Положение на схеме', '02 / 06'] },
  { name: 'Промежуточный слой 3', short: 'Эфирный слой 3', tag: 'Слой 3 · условное обозначение', text: 'Одна из промежуточных эфирных энергий. Отдельное название и характеристики в текстах не приведены; цвет и толщина выбраны для демонстрации.', metric: ['Положение на схеме', '03 / 06'] },
  { name: 'Промежуточный слой 4', short: 'Эфирный слой 4', tag: 'Слой 4 · условное обозначение', text: 'Промежуточная часть эфирной прослойки между кластерами. По тексту, энергии прослойки участвуют в формировании квантовых полей.', metric: ['Положение на схеме', '04 / 06'] },
  { name: 'Промежуточный слой 5', short: 'Эфирный слой 5', tag: 'Слой 5 · условное обозначение', text: 'Одна из промежуточных энергий. Размеры оболочек не отражают измеренные расстояния или реальные пропорции.', metric: ['Положение на схеме', '05 / 06'] },
  { name: 'Граница второго кластера', short: 'Граница кластера 2', tag: 'Слой 6 · у второго кластера', text: 'Внутренняя сторона эфирной прослойки, обращённая ко второму кластеру. Сферы показывают отношения вложенности, а не физическую форму миров.', metric: ['Положение на схеме', '06 / 06'] },
];
export const RAYS = [
  { name: 'Магнитное поле', speed: 0.85, color: '#6fcadf' },
  { name: 'Свет', speed: 1, color: '#f4d395' },
  { name: 'Гравитация', speed: 2.5, color: '#7cddba' },
  { name: 'Свободные гравитоны', speed: 8.25, color: '#a69aef' },
  { name: 'Торсионное поле', speed: 30, color: '#e79acb' },
];
const TAU = Math.PI * 2;
export function rng(seed = 42) { return () => { seed = (1664525 * seed + 1013904223) >>> 0; return seed / 4294967296; }; }
function material(color, opacity = 1) { return new THREE.LineBasicMaterial({ color, transparent: true, opacity, depthWrite: false }); }
function line(parent, points, color, opacity = 1) {
  const obj = new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), material(color, opacity)); parent.add(obj); return obj;
}
function sphere(parent, radius, color, opacity = 1) {
  const obj = new THREE.Mesh(new THREE.SphereGeometry(radius, 20, 14), new THREE.MeshBasicMaterial({ color, transparent: opacity < 1, opacity, depthWrite: opacity === 1 })); parent.add(obj); return obj;
}
function ring(parent, radius, color, opacity = .45, y = 0) {
  return line(parent, Array.from({ length: 129 }, (_, i) => new THREE.Vector3(Math.cos(i / 128 * TAU) * radius, y, Math.sin(i / 128 * TAU) * radius)), color, opacity);
}
function cloud(parent, count, radius, color, seed = 42, size = .035) {
  const random = rng(seed), points = [];
  for (let i = 0; i < count; i++) { const r = radius * Math.cbrt(random()), theta = TAU * random(), z = random() * 2 - 1, xy = Math.sqrt(1 - z*z); points.push(r * xy * Math.cos(theta), r * z, r * xy * Math.sin(theta)); }
  const geometry = new THREE.BufferGeometry(); geometry.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
  const obj = new THREE.Points(geometry, new THREE.PointsMaterial({ color, size, transparent: true, opacity: .65, depthWrite: false, blending: THREE.AdditiveBlending })); parent.add(obj); return obj;
}
function shell(parent, radius, color, cut) {
  const g = new THREE.Group(); parent.add(g);
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(radius, 64, 32, cut ? Math.PI : 0, cut ? Math.PI * 1.5 : TAU), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: .065, side: THREE.DoubleSide, depthWrite: false }));
  g.add(mesh);
  const start = cut ? Math.PI : 0, length = cut ? Math.PI * 1.5 : TAU;
  for (const theta of [Math.PI * .25, Math.PI * .5, Math.PI * .75]) {
    line(g, Array.from({ length: 97 }, (_, i) => { const phi = start + i / 96 * length; return new THREE.Vector3(-radius * Math.cos(phi)*Math.sin(theta), radius*Math.cos(theta), radius*Math.sin(phi)*Math.sin(theta)); }), color, .28);
  }
  for (const phi of cut ? [start, start + length] : [0, Math.PI / 2]) {
    line(g, Array.from({ length: 65 }, (_, i) => { const theta = i / 64 * Math.PI; return new THREE.Vector3(-radius*Math.cos(phi)*Math.sin(theta), radius*Math.cos(theta), radius*Math.sin(phi)*Math.sin(theta)); }), color, .65);
  }
  return { group: g, mesh };
}
export function disposeGroup(group) {
  const geometries = new Set(), materials = new Set();
  group.traverse(obj => { if (obj.geometry) geometries.add(obj.geometry); if (obj.material) (Array.isArray(obj.material) ? obj.material : [obj.material]).forEach(m => materials.add(m)); });
  geometries.forEach(g => g.dispose()); materials.forEach(m => m.dispose()); group.removeFromParent();
}

export function makeLayers(options = {}) {
  const group = new THREE.Group(), shells = [], labels = [], picks = [];
  const core = new THREE.Group(); group.add(core);
  shell(core, 2.45, '#f3c37e', options.cut !== false);
  const quantum = cloud(core, 420, 2.35, '#7fdccf', 27, .035);
  // Three rings identify levels 1–3 without assigning states of matter to them.
  [1.0, 1.5, 2.0].forEach((r, i) => { const orbit = ring(core, r, '#e6bb7d', .5); orbit.rotation.set(i * .7, 0, i * .5); });
  sphere(core, .16, '#ffdda0');
  for (let i = 0; i < 6; i++) {
    const obj = shell(group, 3.0 + i * .47, COLORS[i], options.cut !== false);
    obj.mesh.userData.layer = i; picks.push(obj.mesh); shells.push(obj);
  }
  const outer = shell(group, 6.05, '#687eaa', options.cut !== false);
  outer.mesh.material.opacity = .025;
  const torsion = new THREE.Group(); group.add(torsion);
  for (let k = 0; k < 5; k++) {
    const pts = Array.from({ length: 401 }, (_, i) => {
      const t = i / 400, angle = t * TAU * 12.5, y = (t - .5) * 3.9;
      return new THREE.Vector3(.15 * Math.cos(angle), y, .15 * Math.sin(angle));
    });
    const spiral = line(torsion, pts, '#76edcc', .5); spiral.position.set(Math.cos(k / 5 * TAU) * 2.7, 0, Math.sin(k / 5 * TAU)*2.7);
  }
  labels.push({ text: 'Кластер 1 · уровни 1–3', position: new THREE.Vector3(0, -.3, 2.5), color: '#f3c37e' });
  labels.push({ text: 'Кластер 2 · уровни 4–6', position: new THREE.Vector3(-3.4, 4.5, 0), color: '#a4b1da' });
  labels.push({ text: 'Эфирная прослойка · 6 слоёв', position: new THREE.Vector3(4.4, -3.8, 0), color: '#ba9ceb' });
  return { group, labels, picks, camera: [11, 7, 13], target: [0, 0, 0],
    update(t, state) {
      quantum.rotation.y = t * .065; torsion.rotation.y = t * .06;
      shells.forEach((obj, i) => { obj.group.position.y = state.explode * (i + 1) * .55; obj.mesh.material.opacity = i === state.layer ? .18 : .045; });
      outer.group.position.y = state.explode * 4.2;
      labels[1].position.y = 4.5 + state.explode * 4.2;
      labels[2].position.y = -3.8 + state.explode * 2;
    }
  };
}

export function makeAtom() {
  const group = new THREE.Group(), labels = [], seeds = [], random = rng(73), count = 600;
  const positions = new Float32Array(count * 3), colors = new Float32Array(count * 3);
  const violet = new THREE.Color('#b39aff'), amber = new THREE.Color('#ffd08c');
  for (let i = 0; i < count; i++) {
    const angle = random()*TAU, radius = 4 + random()*2.6, y = (random()-.5)*7;
    seeds.push({ angle, radius, y, target: new THREE.Vector3((random()-.5)*.9, (random()-.5)*.9, (random()-.5)*.9) });
    (i < count * .73 ? violet : amber).toArray(colors, i*3);
  }
  const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(positions,3)); geo.setAttribute('color',new THREE.BufferAttribute(colors,3));
  const particles = new THREE.Points(geo,new THREE.PointsMaterial({ size: .065,vertexColors:true,transparent:true,opacity:.85,depthWrite:false,blending:THREE.AdditiveBlending })); group.add(particles);
  const atom = new THREE.Group(); group.add(atom);
  const nucleus = sphere(atom,.38,'#ffe1a4');
  const electrons = [];
  for(let i=0;i<3;i++) { const g = new THREE.Group(); atom.add(g); g.rotation.set(i*.8, i*.65, i*.5); ring(g,1.75,'#85dddb',.5); electrons.push(sphere(g,.09,'#a7fff0')); }
  const field = cloud(group,500,6,'#6877b7',18,.025);
  for (let i=0;i<5;i++) { const r=ring(group,3+i*.62,'#8b77c8',.16); r.rotation.x=.28*i; }
  labels.push({text:'73% · эфирные поля',position:new THREE.Vector3(-4.2,2.6,0),color:'#b39aff'});
  labels.push({text:'27% · излучение атомов',position:new THREE.Vector3(4.2,-2.7,0),color:'#ffd08c'});
  return {group,labels,picks:[],camera:[9,6,12],target:[0,0,0], update(t,state) {
    const progress = state.assembly, ease = progress*progress*(3-2*progress);
    for(let i=0;i<count;i++) { const s=seeds[i], a=s.angle+t*.12+ease*TAU*1.5, r=s.radius*(1-ease);
      positions[i*3]=Math.cos(a)*r+s.target.x*ease; positions[i*3+1]=s.y*(1-ease)+s.target.y*ease; positions[i*3+2]=Math.sin(a)*r+s.target.z*ease;
    }
    geo.attributes.position.needsUpdate=true;
    atom.scale.setScalar(Math.max(.001,ease)); nucleus.scale.setScalar(1+.04*Math.sin(t*2));
    electrons.forEach((e,i)=>e.position.set(Math.cos(t*1.3+i*2)*1.75,0,Math.sin(t*1.3+i*2)*1.75)); field.rotation.y=t*.025;
  }};
}

export function makeGravity() {
  const group=new THREE.Group(), labels=[], inward=new THREE.Group(), outward=new THREE.Group(); group.add(inward,outward);
  const left=new THREE.Vector3(-3.1,0,0),right=new THREE.Vector3(3.1,0,0);
  const a=sphere(group,.42,'#f1c689');a.position.copy(left);
  const b=sphere(group,.7,'#a9a0ee');b.position.copy(right);
  const pathsA=[],pathsB=[],dotsA=[],dotsB=[];
  function pathA(u,k) { const r=2.15*(1-u)+.18,angle=u*TAU*2+k/6*TAU; return new THREE.Vector3(left.x+Math.cos(angle)*r,(u-.5)*1.7,Math.sin(angle)*r); }
  function pathB(u,k) { const theta=u*TAU,angle=k/7*TAU,r=2.7*Math.sin(theta/2); return new THREE.Vector3(right.x+Math.cos(angle)*r,2.4*Math.cos(theta/2),Math.sin(angle)*r); }
  for(let k=0;k<6;k++) {pathsA.push(line(inward,Array.from({length:161},(_,i)=>pathA(i/160,k)),'#78e2c3',.45)); dotsA.push(sphere(inward,.07,'#b0ffe0'));}
  for(let k=0;k<7;k++) {pathsB.push(line(outward,Array.from({length:161},(_,i)=>pathB(i/160,k)),'#aa9bf0',.45)); dotsB.push(sphere(outward,.07,'#d1c0ff'));}
  const bridge=line(group,Array.from({length:101},(_,i)=>new THREE.Vector3(-2.7+i/100*5.1,Math.sin(i/100*Math.PI)*.35,0)),'#dec9a1',.2);
  labels.push({text:'А · направлено внутрь',position:new THREE.Vector3(-3.1,-2.4,0),color:'#78e2c3'});
  labels.push({text:'Б · поле вокруг объекта',position:new THREE.Vector3(3.1,-3.1,0),color:'#aa9bf0'});
  return {group,labels,picks:[],camera:[7,6,15],target:[0,0,0],update(t,state) {
    inward.visible=state.gravity!=='b';outward.visible=state.gravity!=='a';bridge.visible=state.gravity==='both';
    dotsA.forEach((d,k)=>d.position.copy(pathA((t*.18+k/6)%1,k)));
    dotsB.forEach((d,k)=>d.position.copy(pathB((t*.15+k/7)%1,k)));
  }};
}

export function makeSpeeds() {
  const group=new THREE.Group(),labels=[],dots=[],trails=[],finish=[];
  RAYS.forEach((ray,i)=>{
    const y=3.6-i*1.8;
    line(group,[new THREE.Vector3(-6,y,0),new THREE.Vector3(6,y,0)],'#52677d',.4);
    for(let x=-6;x<=6;x+=2) line(group,[new THREE.Vector3(x,y-.12,0),new THREE.Vector3(x,y+.12,0)],'#52677d',.5);
    const d=sphere(group,.12,ray.color);dots.push(d);
    const trail=line(group,[new THREE.Vector3(-6,y,0),new THREE.Vector3(-6,y,0)],ray.color,.9);trails.push(trail);
    const end=sphere(group,.06,ray.color,.5);end.position.set(6,y,0);finish.push(end);
    labels.push({text:ray.name,position:new THREE.Vector3(-7.25,y,0),color:ray.color});
  });
  return {group,labels,picks:[],camera:[0,1.5,21],target:[-1,0,0],update(t,state) {
    // One model time unit = c traversing the whole track. All rays share start and time.
    RAYS.forEach((r,i)=>{ const f=Math.min(1,state.race*r.speed),y=3.6-i*1.8; dots[i].position.set(-6+12*f,y,0); const a=trails[i].geometry.attributes.position;a.setXYZ(1,-6+12*f,y,0);a.needsUpdate=true;finish[i].scale.setScalar(f===1?2:1); });
  }};
}
