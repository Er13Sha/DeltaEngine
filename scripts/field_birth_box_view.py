"""Кадры двух сцен в шаре и страница three.js. Вердикт не переписывает."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from models.law import seeds
from scripts.field_birth import shell
from scripts.field_birth_box import CORE, march_box

OUT = Path("viz/field-birth-box.html")


def frames_for(with_sea: bool) -> list[list[list[float]]]:
    core = seeds(CORE)
    pos = np.vstack([core, shell(12, 2.0), shell(18, 2.5), shell(24, 3.0)]) if with_sea else core.copy()
    vel = np.zeros_like(pos)
    samples: list[list[list[float]]] = []
    march_box(pos, vel, samples, 0.1)
    return samples


def main() -> None:
    payload = {
        "wall": 4,
        "dt": 0.1,
        "tEnd": 10,
        "empty": frames_for(False),
        "sea": frames_for(True),
    }
    data = json.dumps(payload, separators=(",", ":"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(PAGE.replace("/*DATA*/", data), encoding="utf-8")
    print(
        f"empty={len(payload['empty'])} sea={len(payload['sea'])} bytes={OUT.stat().st_size}",
        flush=True,
    )
    print(f"wrote {OUT}", flush=True)


PAGE = r"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <title>Пятно в шаре радиуса 4</title>
  <style>
    html, body { margin: 0; height: 100%; background: #000; color: #fff; font: 14px/1.35 "Courier New", monospace; }
    #view { position: fixed; inset: 0; }
    #panel {
      position: fixed; left: 12px; top: 12px; z-index: 2;
      width: 420px; text-shadow: 2px 2px 4px #000;
    }
    h1 { font-size: 15px; font-weight: 600; margin: 0 0 8px; }
    p { margin: 0 0 8px; }
    label { display: block; margin: 8px 0 4px; }
    button, select, input[type="range"] { font: inherit; }
    button, select { background: transparent; color: #fff; border: 1px solid #444; padding: 4px 8px; }
    .row { display: flex; gap: 8px; align-items: center; }
    input[type="range"] { width: 100%; }
    .swatch { display: inline-block; width: 10px; height: 10px; margin-right: 6px; vertical-align: -1px; }
  </style>
</head>
<body>
  <div id="view"></div>
  <div id="panel">
    <h1>Пятно в шаре радиуса 4</h1>
    <p>Кадры прогона в шаре радиуса 4. Закон 1/r², P = 0. Рисунок частицы — тот же спрайт, что в scripts/indexGPU.html: круг, степень 1,6, сложение света. Размер задан номером кванта. Цвет — скорость между кадрами, от синего к золотому. Ни размер, ни цвет в силу не входят. Масса у всех квантов одна.</p>
    <label>Сцена
      <select id="scene">
        <option value="empty">Пустая, 8 квантов</option>
        <option value="sea" selected>Среда, 62 кванта</option>
      </select>
    </label>
    <div class="row">
      <button id="play" type="button">Пауза</button>
      <span id="clock">t = 0</span>
    </div>
    <label>Время
      <input id="time" type="range" min="0" max="100" value="0" step="1" />
    </label>
    <p id="legend"></p>
  </div>
  <script type="importmap">
  {
    "imports": {
      "three": "https://unpkg.com/three@0.170.0/build/three.module.js",
      "three/addons/": "https://unpkg.com/three@0.170.0/examples/jsm/"
    }
  }
  </script>
  <script type="module">
    import * as THREE from "three";
    import { OrbitControls } from "three/addons/controls/OrbitControls.js";

    const DATA = /*DATA*/;
    const GROUPS = {
      empty: [{ name: "ядро", from: 0, to: 8, color: 0xf2c14e }],
      sea: [
        { name: "ядро", from: 0, to: 8, color: 0xf2c14e },
        { name: "внутренняя оболочка", from: 8, to: 20, color: 0x7eb6ff },
        { name: "средняя оболочка", from: 20, to: 38, color: 0x8fd18d },
        { name: "внешняя оболочка", from: 38, to: 62, color: 0xd0d4dc }
      ]
    };

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    document.getElementById("view").appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);
    const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 80);
    camera.position.set(9, 7, 11);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0);
    controls.update();

    const wall = new THREE.Mesh(
      new THREE.SphereGeometry(DATA.wall, 48, 32),
      new THREE.MeshBasicMaterial({ color: 0x223044, wireframe: true, transparent: true, opacity: 0.08 })
    );
    scene.add(wall);

    const quad = new THREE.PlaneGeometry(1, 1);
    function spriteMaterial() {
      return new THREE.ShaderMaterial({
        vertexShader: `
          attribute float aHeat;
          varying vec2 vUv;
          varying vec3 vColor;
          void main() {
            vUv = uv;
            float h = clamp(aHeat, 0.0, 2.0);
            float mm = 0.0;
            vec3 cold = vec3(0.10, 0.35, 0.85);
            vec3 hot = vec3(1.00, 0.75, 0.30);
            vec3 heavy = vec3(0.95, 0.35, 0.95);
            vec3 c = mix(cold, hot, clamp(h, 0.0, 1.0));
            vColor = mix(c, heavy, mm * 0.55);
            vec4 mv = modelViewMatrix * instanceMatrix * vec4(position, 1.0);
            gl_Position = projectionMatrix * mv;
          }
        `,
        fragmentShader: `
          precision mediump float;
          varying vec2 vUv;
          varying vec3 vColor;
          void main() {
            float d = distance(vUv, vec2(0.5));
            if (d > 0.5) discard;
            float a = pow(1.0 - d * 2.0, 1.6);
            gl_FragColor = vec4(vColor, a);
          }
        `,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending
      });
    }

    function dipoleLength(i) {
      return 0.28 + 0.62 * ((i * 7 + 3) % 11) / 10;
    }

    let meshes = [];
    let frames = DATA.sea;
    let frame = 0;
    let playing = true;

    function clearMeshes() {
      for (const mesh of meshes) {
        scene.remove(mesh);
        mesh.geometry.dispose();
        mesh.material.dispose();
      }
      meshes = [];
    }

    function build(name) {
      clearMeshes();
      frames = DATA[name];
      const count = frames[0].length;
      const geometry = quad.clone();
      geometry.setAttribute("aHeat", new THREE.InstancedBufferAttribute(new Float32Array(count), 1));
      const mesh = new THREE.InstancedMesh(geometry, spriteMaterial(), count);
      scene.add(mesh);
      meshes.push(mesh);
      const legend = document.getElementById("legend");
      legend.replaceChildren();
      const cold = document.createElement("div");
      cold.append(swatch("#1a59d9"), document.createTextNode("медленный кадр"));
      const hot = document.createElement("div");
      hot.append(swatch("#ffbf4d"), document.createTextNode("быстрый кадр"));
      legend.append(cold, hot);
      for (const group of GROUPS[name]) {
        const line = document.createElement("div");
        line.textContent = group.name + " · " + (group.to - group.from);
        legend.append(line);
      }
      frame = Math.min(frame, frames.length - 1);
      show(frame);
    }

    function swatch(color) {
      const mark = document.createElement("span");
      mark.className = "swatch";
      mark.style.background = color;
      return mark;
    }

    const dummy = new THREE.Object3D();
    function show(index) {
      const positions = frames[index];
      const next = frames[Math.min(frames.length - 1, index + 1)];
      const mesh = meshes[0];
      const heatAttr = mesh.geometry.getAttribute("aHeat");
      for (let i = 0; i < positions.length; i += 1) {
        const p = positions[i];
        const q = next[i];
        const speed = Math.hypot(q[0] - p[0], q[1] - p[1], q[2] - p[2]) / DATA.dt;
        const heat = Math.min(2, speed / 3);
        heatAttr.setX(i, heat);
        dummy.position.set(p[0], p[1], p[2]);
        dummy.quaternion.copy(camera.quaternion);
        dummy.scale.setScalar(dipoleLength(i) * (1 + heat * 0.45) * 1.6);
        dummy.updateMatrix();
        mesh.setMatrixAt(i, dummy.matrix);
      }
      heatAttr.needsUpdate = true;
      mesh.instanceMatrix.needsUpdate = true;
      const t = (index * DATA.dt).toFixed(1).replace(".", ",");
      document.getElementById("clock").textContent = "t = " + t;
      document.getElementById("time").value = String(index);
      document.getElementById("time").max = String(frames.length - 1);
    }

    document.getElementById("scene").addEventListener("change", (event) => build(event.target.value));
    document.getElementById("play").addEventListener("click", () => {
      playing = !playing;
      document.getElementById("play").textContent = playing ? "Пауза" : "Пуск";
    });
    document.getElementById("time").addEventListener("input", (event) => {
      playing = false;
      document.getElementById("play").textContent = "Пуск";
      show(Number(event.target.value));
    });
    window.addEventListener("resize", () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    });

    build("sea");
    let last = 0;
    function tick(now) {
      requestAnimationFrame(tick);
      if (playing && now - last > 70) {
        last = now;
        frame = (frame + 1) % frames.length;
      }
      show(frame);
      controls.update();
      renderer.render(scene, camera);
    }
    requestAnimationFrame(tick);
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
