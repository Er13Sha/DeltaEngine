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
    html, body { margin: 0; height: 100%; background: #121418; color: #e6e6e6; font: 14px/1.4 sans-serif; }
    #view { position: fixed; inset: 0; }
    #panel {
      position: fixed; left: 16px; top: 16px; z-index: 2;
      background: rgba(18, 20, 24, 0.88); border: 1px solid #2c313a;
      padding: 12px 14px; width: 280px;
    }
    h1 { font-size: 15px; font-weight: 600; margin: 0 0 8px; }
    p { margin: 0 0 8px; color: #b7bdc7; }
    label { display: block; margin: 8px 0 4px; }
    button, select, input[type="range"] { font: inherit; }
    button, select { background: #1c212b; color: #e6e6e6; border: 1px solid #3a4150; padding: 4px 8px; }
    .row { display: flex; gap: 8px; align-items: center; }
    input[type="range"] { width: 100%; }
    .swatch { display: inline-block; width: 10px; height: 10px; margin-right: 6px; vertical-align: -1px; }
  </style>
</head>
<body>
  <div id="view"></div>
  <div id="panel">
    <h1>Пятно в шаре радиуса 4</h1>
    <p>Закон между квантами 1/r², P = 0. Стенка отражает вышедших за радиус 4. Кадры каждые 0,1 до t = 10.</p>
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
      empty: [{ name: "середина", from: 0, to: 8, color: 0xf2c14e }],
      sea: [
        { name: "середина", from: 0, to: 8, color: 0xf2c14e },
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
    scene.background = new THREE.Color(0x121418);
    const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 80);
    camera.position.set(9, 7, 11);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0);
    controls.update();

    scene.add(new THREE.AmbientLight(0xffffff, 0.65));
    const key = new THREE.DirectionalLight(0xffffff, 1.1);
    key.position.set(6, 10, 4);
    scene.add(key);

    const wall = new THREE.Mesh(
      new THREE.SphereGeometry(DATA.wall, 48, 32),
      new THREE.MeshBasicMaterial({ color: 0x8aa0c8, wireframe: true, transparent: true, opacity: 0.35 })
    );
    scene.add(wall);

    const dot = new THREE.SphereGeometry(0.12, 16, 12);
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
      const groups = GROUPS[name];
      const legend = document.getElementById("legend");
      legend.replaceChildren();
      for (const group of groups) {
        const count = group.to - group.from;
        const material = new THREE.MeshStandardMaterial({ color: group.color, roughness: 0.45 });
        const mesh = new THREE.InstancedMesh(dot, material, count);
        mesh.userData.group = group;
        scene.add(mesh);
        meshes.push(mesh);
        const line = document.createElement("div");
        const mark = document.createElement("span");
        mark.className = "swatch";
        mark.style.background = "#" + group.color.toString(16).padStart(6, "0");
        line.append(mark, document.createTextNode(group.name + " · " + count));
        legend.append(line);
      }
      frame = Math.min(frame, frames.length - 1);
      show(frame);
    }

    const dummy = new THREE.Object3D();
    function show(index) {
      const positions = frames[index];
      for (const mesh of meshes) {
        const group = mesh.userData.group;
        for (let i = group.from; i < group.to; i += 1) {
          const p = positions[i];
          dummy.position.set(p[0], p[1], p[2]);
          dummy.updateMatrix();
          mesh.setMatrixAt(i - group.from, dummy.matrix);
        }
        mesh.instanceMatrix.needsUpdate = true;
      }
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
        show(frame);
      }
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
