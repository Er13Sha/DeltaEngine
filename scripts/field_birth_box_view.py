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
    html, body { margin: 0; height: 100%; background: #07090d; color: #e6e6e6; font: 14px/1.4 sans-serif; }
    #view { position: fixed; inset: 0; }
    #panel {
      position: fixed; left: 16px; top: 16px; z-index: 2;
      background: rgba(7, 9, 13, 0.88); border: 1px solid #2c313a;
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
    <p>Закон между квантами 1/r², P = 0. Стенка отражает вышедших за радиус 4. Квант рисует фрагментный шейдер: два мягких пятна, голова крупнее хвоста. Длина задана номером и в силу не входит. Кадры каждые 0,1 до t = 10.</p>
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
    scene.background = new THREE.Color(0x07090d);
    const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 80);
    camera.position.set(9, 7, 11);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0);
    controls.update();

    const wall = new THREE.Mesh(
      new THREE.SphereGeometry(DATA.wall, 48, 32),
      new THREE.MeshBasicMaterial({ color: 0x8aa0c8, wireframe: true, transparent: true, opacity: 0.35 })
    );
    scene.add(wall);

    const quad = new THREE.PlaneGeometry(1, 1);
    function spriteMaterial(color, gain) {
      return new THREE.ShaderMaterial({
        uniforms: {
          uColor: { value: new THREE.Color(color) },
          uGain: { value: gain }
        },
        vertexShader: `
          varying vec2 vUv;
          void main() {
            vUv = uv;
            vec4 mv = modelViewMatrix * instanceMatrix * vec4(position, 1.0);
            gl_Position = projectionMatrix * mv;
          }
        `,
        fragmentShader: `
          precision mediump float;
          varying vec2 vUv;
          uniform vec3 uColor;
          uniform float uGain;
          void main() {
            vec2 p = vUv * 2.0 - 1.0;
            float r2 = dot(p, p);
            if (r2 > 1.0) discard;
            float core = exp(-r2 * 7.0);
            float halo = exp(-r2 * 1.7);
            float alpha = clamp((core + halo * 0.28) * uGain, 0.0, 1.0);
            gl_FragColor = vec4(uColor * (0.25 + core) * uGain, alpha);
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
    function dipoleAxis(i, target) {
      const yaw = ((i * 17) % 360) * Math.PI / 180;
      const pitch = ((i * 29) % 180) * Math.PI / 180;
      target.set(
        Math.sin(pitch) * Math.cos(yaw),
        Math.cos(pitch),
        Math.sin(pitch) * Math.sin(yaw)
      );
    }

    let meshes = [];
    let frames = DATA.sea;
    let frame = 0;
    let playing = true;

    function clearMeshes() {
      for (const mesh of meshes) {
        scene.remove(mesh);
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
        const head = new THREE.InstancedMesh(quad, spriteMaterial(group.color, 1.0), count);
        const tail = new THREE.InstancedMesh(quad, spriteMaterial(group.color, 0.55), count);
        head.userData.group = group;
        head.userData.pole = 1;
        tail.userData.group = group;
        tail.userData.pole = -1;
        scene.add(head);
        scene.add(tail);
        meshes.push(head, tail);
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
    const axis = new THREE.Vector3();
    function show(index) {
      const positions = frames[index];
      for (const mesh of meshes) {
        const group = mesh.userData.group;
        const pole = mesh.userData.pole;
        for (let i = group.from; i < group.to; i += 1) {
          const p = positions[i];
          const length = dipoleLength(i);
          dipoleAxis(i, axis);
          const radius = (pole > 0 ? length * 0.42 : length * 0.22) * 2.6;
          dummy.position.set(
            p[0] + axis.x * pole * length * 0.45,
            p[1] + axis.y * pole * length * 0.45,
            p[2] + axis.z * pole * length * 0.45
          );
          dummy.quaternion.copy(camera.quaternion);
          dummy.scale.setScalar(radius);
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
