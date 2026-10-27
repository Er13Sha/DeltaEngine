import assert from 'node:assert/strict';
import { makeLayers, makeAtom, makeGravity, makeSpeeds, disposeGroup, RAYS } from './src/model.js';

const state={explode:0,layer:0,assembly:0,gravity:'both',race:0};
let scenarios=0;
function finite(model) {
  model.group.updateMatrixWorld(true);
  model.group.traverse(obj=>{
    assert.ok(obj.matrixWorld.elements.every(Number.isFinite),'Object transform must be finite');
    const positions=obj.geometry?.attributes.position;
    if(positions) assert.ok(positions.array.every(Number.isFinite),'Geometry positions must be finite');
  });
  scenarios++;
}
for(const cut of [true,false]){
  const model=makeLayers({cut});assert.equal(model.picks.length,6);
  for(const explode of [0,.5,1])for(let layer=0;layer<6;layer++){model.update(9,{...state,explode,layer});finite(model);}
  disposeGroup(model.group);
}
const atom=makeAtom();
const particles=atom.group.children.find(obj=>obj.isPoints&&obj.geometry.attributes.color);
assert.equal(particles.geometry.attributes.position.count,600);
const colors=particles.geometry.attributes.color;
let violet=0;for(let i=0;i<colors.count;i++)if(colors.getZ(i)>colors.getX(i))violet++;
assert.equal(violet,438,'73% of 600 particles come from the ether field');
for(const assembly of [0,.25,.5,.75,1]){atom.update(7,{...state,assembly});finite(atom);}
const positions=particles.geometry.attributes.position;
for(let i=0;i<positions.count;i++)assert.ok(Math.hypot(positions.getX(i),positions.getY(i),positions.getZ(i))<.8,'Particles converge into nucleus');
disposeGroup(atom.group);
const gravity=makeGravity();
for(const mode of ['both','a','b']){gravity.update(12,{...state,gravity:mode});finite(gravity);assert.equal(gravity.group.children[0].visible,mode!=='b');assert.equal(gravity.group.children[1].visible,mode!=='a');}
disposeGroup(gravity.group);
const speeds=makeSpeeds();
const dots=speeds.group.children.filter(obj=>obj.isMesh&&obj.geometry.parameters.radius===.12);
assert.equal(dots.length,5);
for(const race of [0,.01,.02,.2,1,1/.85]){
  speeds.update(0,{...state,race});finite(speeds);
  dots.forEach((dot,i)=>assert.ok(Math.abs(dot.position.x-(-6+12*Math.min(1,race*RAYS[i].speed)))<1e-8,'Speeds share a linear time scale'));
}
dots.forEach(dot=>assert.equal(dot.position.x,6,'All pulses reach the finish'));
disposeGroup(speeds.group);
console.log(`PASS: ${scenarios} geometry states; 6 layers; 73/27 composition; A/B visibility; linear speeds and finish.`);
