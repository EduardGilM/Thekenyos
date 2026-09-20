import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
const $ = id => document.getElementById(id);

// ---------- renderer / scene ----------
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(devicePixelRatio, 1.75));
renderer.setSize(innerWidth, innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
document.body.prepend(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x9fc4e6);
scene.fog = new THREE.Fog(0xb8d3ea, 30, 110);
const camera = new THREE.PerspectiveCamera(70, innerWidth / innerHeight, .05, 300);
camera.up.set(0, 0, 1);

const hemi = new THREE.HemisphereLight(0xcfe6ff, 0x5f7a3a, 1.1);
scene.add(hemi);
const sun = new THREE.DirectionalLight(0xfff1d6, 2.4);
sun.castShadow = true;
sun.shadow.mapSize.set(4096, 4096);
sun.shadow.camera.near = 1; sun.shadow.camera.far = 80;
sun.shadow.bias = -.0006; sun.shadow.normalBias = .02;
scene.add(sun, sun.target);

// ---------- helpers ----------
const fetchBin = async u => {
  const r = await fetch(u);
  if (r.ok && !(r.headers.get('content-type') || '').startsWith('text/')) return new Uint8Array(await r.arrayBuffer());
  // hosts that only serve text: the same bytes, base64 in a .txt
  const b64 = (await (await fetch(u + '.txt')).text()).trim();
  const bin = atob(b64), out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
};
const f32 = (buf, off, n) => new Float32Array(buf.buffer, buf.byteOffset + off, n);
const u32 = (buf, off, n) => new Uint32Array(buf.buffer, buf.byteOffset + off, n);
const mjQuat = (w, x, y, z) => new THREE.Quaternion(x, y, z, w);

// ---------- orchard ----------
let terrain;
async function loadOrchard() {
  const meta = await (await fetch('orchard/orchard.json')).json();
  const bin = await fetchBin('orchard/orchard.bin');
  const half = meta.half, n = meta.terrain.n;
  const H = f32(bin, 0, n * n);
  terrain = { n, half, H, sample(x, y) {
    const u = (x + half) / (2 * half) * (n - 1), v = (y + half) / (2 * half) * (n - 1);
    const i = Math.max(0, Math.min(n - 2, Math.floor(u))), j = Math.max(0, Math.min(n - 2, Math.floor(v)));
    const fu = Math.min(1, Math.max(0, u - i)), fv = Math.min(1, Math.max(0, v - j));
    const h00 = H[j * n + i], h10 = H[j * n + i + 1], h01 = H[(j + 1) * n + i], h11 = H[(j + 1) * n + i + 1];
    return (h00 * (1 - fu) + h10 * fu) * (1 - fv) + (h01 * (1 - fu) + h11 * fu) * fv;
  } };
  // ground: heightfield plane (rows = y, cols = x)
  const g = new THREE.PlaneGeometry(2 * half, 2 * half, n - 1, n - 1);
  const pos = g.attributes.position;
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const k = j * n + i; // PlaneGeometry rows go from +y to -y
    pos.setXYZ(k, -half + i * 2 * half / (n - 1), half - j * 2 * half / (n - 1), H[(n - 1 - j) * n + i]);
  }
  g.computeVertexNormals();
  const tex = new THREE.TextureLoader().load('orchard/ground.jpg');
  tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 8; tex.flipY = false;
  const ground = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ map: tex, roughness: 1 }));
  ground.receiveShadow = true;
  scene.add(ground);
  // earth skirt so the field is not a floating card
  const skirt = new THREE.Mesh(new THREE.BoxGeometry(2 * half + 1.6, 2 * half + 1.6, 3),
    new THREE.MeshStandardMaterial({ color: 0x4a3524, roughness: 1 }));
  skirt.position.z = meta.terrain.min_z - 1.5 - .01;
  scene.add(skirt);
  // far ground beyond the block
  const far = new THREE.Mesh(new THREE.PlaneGeometry(600, 600), new THREE.MeshStandardMaterial({ color: 0x3f6a26, roughness: 1 }));
  far.position.z = meta.terrain.min_z - .02; scene.add(far);

  // wood: posts, wires, canes as merged cylinders
  const wood = f32(bin, meta.wood.off, meta.wood.n * 8);
  const woodCols = [new THREE.Color(0.45, 0.28, 0.12), new THREE.Color(0.35, 0.22, 0.10), new THREE.Color(0.28, 0.42, 0.14)];
  const parts = [];
  const a = new THREE.Vector3(), b = new THREE.Vector3(), d = new THREE.Vector3(), q = new THREE.Quaternion(), zAxis = new THREE.Vector3(0, 0, 1);
  const posts = [];
  for (let s = 0; s < meta.wood.n; s++) {
    const o = s * 8;
    a.set(wood[o], wood[o + 1], wood[o + 2]); b.set(wood[o + 3], wood[o + 4], wood[o + 5]);
    const r = wood[o + 6], order = wood[o + 7];
    d.subVectors(b, a); const len = d.length(); if (len < 1e-4) continue;
    const cg = new THREE.CylinderGeometry(r, r, len, order === 0 ? 12 : 6, 1);
    cg.rotateX(Math.PI / 2); // cylinder along z
    q.setFromUnitVectors(zAxis, d.clone().normalize());
    const m = new THREE.Matrix4().compose(a.clone().lerp(b, .5), q, new THREE.Vector3(1, 1, 1));
    cg.applyMatrix4(m);
    const c = woodCols[order]; const col = new Float32Array(cg.attributes.position.count * 3);
    for (let i = 0; i < col.length; i += 3) { col[i] = c.r; col[i + 1] = c.g; col[i + 2] = c.b; }
    cg.setAttribute('color', new THREE.BufferAttribute(col, 3));
    parts.push(cg);
    if (order === 0) posts.push({ x: a.x, y: a.y, r: r + .25 });
  }
  const woodMesh = new THREE.Mesh(mergeGeometries(parts, false), new THREE.MeshStandardMaterial({ vertexColors: true, roughness: .95 }));
  woodMesh.castShadow = true; woodMesh.receiveShadow = true; scene.add(woodMesh);

  // leaves: 3 blade classes, instanced
  const inst = f32(bin, meta.leaves.off, meta.leaves.n * 8);
  const counts = [0, 0, 0]; for (let i = 0; i < meta.leaves.n; i++) counts[inst[i * 8 + 7]]++;
  const leafMat = new THREE.MeshStandardMaterial({ color: new THREE.Color(...meta.leaf_color).multiplyScalar(1.15), side: THREE.DoubleSide, roughness: .8, vertexColors: false });
  const mats = new THREE.Matrix4(), sc = new THREE.Vector3(1, 1, 1), p = new THREE.Vector3();
  const leafMeshes = meta.blades.map((bl, c) => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(f32(bin, bl.voff, bl.nv * 3).slice(), 3));
    geo.setIndex(new THREE.BufferAttribute(u32(bin, bl.foff, bl.nf * 3).slice(), 1));
    geo.computeVertexNormals();
    const im = new THREE.InstancedMesh(geo, leafMat, counts[c]);
    im.castShadow = true; im.receiveShadow = true;
    scene.add(im); return im;
  });
  const cursor = [0, 0, 0];
  const colAttr = leafMeshes.map(im => new Float32Array(im.count * 3));
  for (let i = 0; i < meta.leaves.n; i++) {
    const o = i * 8, c = inst[o + 7];
    p.set(inst[o], inst[o + 1], inst[o + 2]); q.set(inst[o + 3], inst[o + 4], inst[o + 5], inst[o + 6]);
    mats.compose(p, q, sc);
    const k = cursor[c]++;
    leafMeshes[c].setMatrixAt(k, mats);
    const v = .8 + .4 * Math.random(); colAttr[c][k * 3] = v * .95; colAttr[c][k * 3 + 1] = v; colAttr[c][k * 3 + 2] = v * .9;
  }
  leafMeshes.forEach((im, c) => { im.instanceColor = new THREE.InstancedBufferAttribute(colAttr[c], 3); im.instanceMatrix.needsUpdate = true; });

  // Perimeter hedge: an opaque wall of foliage so the flat world has no visible edge.
  {
    const R = half + .9, HGT = 1.4, THK = .8;
    const wallMat = new THREE.MeshStandardMaterial({ color: 0x4a2c12, roughness: 1 });
    const walls = [[R, 0, THK, 2 * R + THK], [-R, 0, THK, 2 * R + THK], [0, R, 2 * R + THK, THK], [0, -R, 2 * R + THK, THK]];
    for (const [x, y, sx, sy] of walls) {
      const w = new THREE.Mesh(new THREE.BoxGeometry(sx, sy, HGT), wallMat);
      w.position.set(x, y, HGT / 2 - .1); w.receiveShadow = true; scene.add(w);
    }
    const blade = leafMeshes[2].geometry;
    const hedgeMat = new THREE.MeshStandardMaterial({ color: 0xffffff, side: THREE.DoubleSide, roughness: .85 });
    const autumn = [[.66, .34, .10], [.78, .48, .14], [.56, .24, .08], [.84, .60, .20], [.48, .30, .12]];
    const perWall = 2400, hedgeLeaves = new THREE.InstancedMesh(blade, hedgeMat, 4 * perWall);
    hedgeLeaves.castShadow = true;
    const cols = new Float32Array(4 * perWall * 3);
    const e = new THREE.Euler(), s2 = new THREE.Vector3();
    let k = 0;
    for (let wi = 0; wi < 4; wi++) for (let i = 0; i < perWall; i++) {
      const along = (Math.random() * 2 - 1) * R, hgt = Math.random() * HGT, depth = -THK / 2 - .12 - Math.random() * .3;
      const scale = 1.4 + Math.random() * 1.2;
      if (wi === 0) p.set(R + depth, along, hgt); else if (wi === 1) p.set(-R - depth, along, hgt);
      else if (wi === 2) p.set(along, R + depth, hgt); else p.set(along, -R - depth, hgt);
      // blade +Z faces roughly into the field, with plenty of scatter
      const face = wi === 0 ? Math.PI : wi === 1 ? 0 : wi === 2 ? -Math.PI / 2 : Math.PI / 2;
      e.set(Math.PI / 2 + (Math.random() - .5) * 1.4, (Math.random() - .5) * 1.2, face + (Math.random() - .5) * 1.6);
      q.setFromEuler(e); s2.setScalar(scale);
      mats.compose(p, q, s2); hedgeLeaves.setMatrixAt(k, mats);
      const c = autumn[Math.floor(Math.random() * autumn.length)], v = .75 + .45 * Math.random(); cols[k * 3] = c[0] * v; cols[k * 3 + 1] = c[1] * v; cols[k * 3 + 2] = c[2] * v; k++;
    }
    hedgeLeaves.instanceColor = new THREE.InstancedBufferAttribute(cols, 3);
    scene.add(hedgeLeaves);
  }

  // canopy kiwis (visual) + stems
  const fr = f32(bin, meta.fruit.off, meta.fruit.n * 12);
  const kiwiGeo = new THREE.SphereGeometry(1, 14, 10);
  const kiwiMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: .9 });
  const kiwis = new THREE.InstancedMesh(kiwiGeo, kiwiMat, meta.fruit.n);
  kiwis.castShadow = true;
  const stemPts = [];
  for (let i = 0; i < meta.fruit.n; i++) {
    const o = i * 12;
    p.set(fr[o], fr[o + 1], fr[o + 2]); sc.set(fr[o + 3], fr[o + 4], fr[o + 5]);
    mats.compose(p, new THREE.Quaternion(), sc); kiwis.setMatrixAt(i, mats);
    kiwis.setColorAt(i, new THREE.Color(fr[o + 6], fr[o + 7], fr[o + 8]));
    stemPts.push(fr[o], fr[o + 1], fr[o + 2] + fr[o + 5], fr[o + 9], fr[o + 10], fr[o + 11]);
  }
  scene.add(kiwis);
  const sg = new THREE.BufferGeometry(); sg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(stemPts), 3));
  scene.add(new THREE.LineSegments(sg, new THREE.LineBasicMaterial({ color: 0x5a4a2a })));
  return { meta, posts };
}

// ---------- robots ----------
const GEOM = { 4: 'ellipsoid', 6: 'box', 7: 'mesh' };
async function loadRobot() {
  const meta = await (await fetch('robot/meta.json')).json();
  const mbin = await fetchBin('robot/meshes.bin');
  let folders = ['robot'];
  try { const idx = await fetch('rollouts/index.json'); if (idx.ok) folders = (await idx.json()).rollouts; } catch (e) {}
  const rollouts = await Promise.all(folders.map(async f => {
    const m = f === 'robot' ? meta : await (await fetch(f + '/meta.json')).json();
    const poses = f32(await fetchBin(f + '/poses.f32'), 0, m.frames * m.bodies.length * 7);
    return { meta: m, poses, dur: m.frames / m.fps };
  }));
  const poses = rollouts[0].poses;
  const geos = {};
  for (const m of meta.meshes) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(f32(mbin, m.voff, m.nv * 3).slice(), 3));
    g.setIndex(new THREE.BufferAttribute(u32(mbin, m.foff, m.nf * 3).slice(), 1));
    g.computeVertexNormals();
    geos[m.id] = g;
  }
  const box = new THREE.BoxGeometry(2, 2, 2), sph = new THREE.SphereGeometry(1, 20, 14);
  const matCache = {};
  const matFor = rgba => { const k = rgba.map(v => v.toFixed(2)).join(); return matCache[k] ??= new THREE.MeshStandardMaterial({ color: new THREE.Color(rgba[0], rgba[1], rgba[2]), roughness: .55, metalness: .15, transparent: rgba[3] < 1, opacity: rgba[3], side: THREE.DoubleSide }); };
  const stemMat = new THREE.LineBasicMaterial({ color: 0x6b5a33 });
  function build() {
    const root = new THREE.Group();
    const bodies = meta.bodies.map(() => { const g = new THREE.Group(); root.add(g); return g; });
    for (const ge of meta.geoms) {
      let mesh;
      if (GEOM[ge.type] === 'mesh') mesh = new THREE.Mesh(geos[ge.mesh], matFor(ge.rgba));
      else if (GEOM[ge.type] === 'box') { mesh = new THREE.Mesh(box, matFor(ge.rgba)); mesh.scale.set(...ge.size); }
      else { mesh = new THREE.Mesh(sph, matFor(ge.rgba)); mesh.scale.set(...ge.size); }
      mesh.position.set(...ge.pos); mesh.quaternion.copy(mjQuat(...ge.quat));
      mesh.castShadow = true; mesh.receiveShadow = true;
      bodies[ge.body].add(mesh);
    }
    // stems for the robot's own kiwis: anchor above the initial fruit position
    return { root, bodies };
  }
  function stemsFor(root, poses) {
    return meta.fruit_bodies.map(fb => {
      const o = fb * 7; const anchor = new THREE.Vector3(poses[o], poses[o + 1], 1.62);   // stem runs up to the leaf roof
      const g = new THREE.BufferGeometry().setFromPoints([anchor, anchor.clone()]);
      const line = new THREE.Line(g, stemMat); root.add(line); return { line, anchor, fb };
    });
  }
  return { meta, poses, build, stemsFor, rollouts };
}

// ---------- world ----------
const state = { t: 0, speed: 1.5, robots: [], follow: -1, playing: false };
const player = { pos: new THREE.Vector3(0, -8, 0), yaw: Math.PI / 2, pitch: 0, eye: 1.32, vel: new THREE.Vector3() };
const keys = {};
window.orchard = { state, player, camera, keys };

async function init() {
  const [orch, rob] = await Promise.all([loadOrchard(), loadRobot()]);
  const { meta } = rob;
  const nb = meta.bodies.length;
  // path centre of the recorded rollout (chassis walks along +x at y ≈ -1.25)
  const cx = 0.9, cy = -1.25;
  const cz = 0;
  // Five robots in different aisles, alternating heading, staggered in time.
  const placements = [
    { x: 0, y: 0, yaw: 0, t0: 0 },
    { x: -5, y: 5, yaw: Math.PI, t0: 61 },
    { x: 5, y: -5, yaw: 0, t0: 122 },
    { x: -10, y: -10, yaw: Math.PI / 2, t0: 30 },
    { x: 10, y: 10, yaw: -Math.PI / 2, t0: 150 },
  ];
  placements.forEach((pl, i) => {
    const ro = rob.rollouts[i % rob.rollouts.length];
    const r = rob.build(); r.stems = rob.stemsFor(r.root, ro.poses); r.ro = ro;
    const gz = terrain.sample(pl.x, pl.y) - cz;
    r.root.position.set(pl.x, pl.y, gz);
    r.root.rotation.z = pl.yaw;
    // shift so the recorded path centre lands on the placement point
    const off = new THREE.Group(); off.position.set(-cx, -cy, 0);
    r.root.add(off); r.bodies.forEach(b => off.add(b)); r.stems.forEach(s => off.add(s.line));
    scene.add(r.root);
    state.robots.push({ ...r, t0: pl.t0, id: i + 1, world: new THREE.Vector3(pl.x, pl.y, gz) });
  });
  $('robots').textContent = `${placements.length} robots · ${meta.fruits} kiwis each · ${rob.rollouts.length} recorded runs`;
  const dur = Math.max(...rob.rollouts.map(r => r.dur));

  const tmpQ = new THREE.Quaternion(), tmpQ2 = new THREE.Quaternion(), tmpP = new THREE.Vector3(), tmpP2 = new THREE.Vector3();
  let frameDt = 1 / 60;
  function poseRobots(t) {
    for (const r of state.robots) {
      const poses = r.ro.poses, rmeta = r.ro.meta, rdur = r.ro.dur;
      const lt = ((t + r.t0) % rdur + rdur) % rdur;
      const f = lt * rmeta.fps, i0 = Math.min(rmeta.frames - 1, Math.floor(f)), i1 = Math.min(rmeta.frames - 1, i0 + 1), a = f - i0;
      for (let b = 0; b < nb; b++) {
        const o0 = (i0 * nb + b) * 7, o1 = (i1 * nb + b) * 7;
        tmpP.set(poses[o0], poses[o0 + 1], poses[o0 + 2]); tmpP2.set(poses[o1], poses[o1 + 1], poses[o1 + 2]);
        tmpQ.set(poses[o0 + 4], poses[o0 + 5], poses[o0 + 6], poses[o0 + 3]); tmpQ2.set(poses[o1 + 4], poses[o1 + 5], poses[o1 + 6], poses[o1 + 3]);
        r.bodies[b].position.copy(tmpP.lerp(tmpP2, a)); r.bodies[b].quaternion.copy(tmpQ.slerp(tmpQ2, a));
      }
      // A kiwi still on its stem hangs still until the robot arrives: the recorded free body
      // jitters and spins at reset, which reads as a bug rather than physics.
      const ch = r.bodies[meta.chassis].position;
      for (const s of r.stems) {
        const fp = r.bodies[s.fb].position;
        const o0 = s.fb * 7; const onStem = Math.hypot(fp.x - poses[o0], fp.y - poses[o0 + 1], fp.z - poses[o0 + 2]) < .25;
        if (onStem) {
          // Hanging fruit: follow the recording while the robot works on it, otherwise ease back
          // to rest so a nudged kiwi does not swing for ten seconds (the sim has no stem damping).
          const near = Math.hypot(ch.x - s.anchor.x, ch.y - s.anchor.y) < 1.15;
          s.disp ??= new THREE.Vector3(poses[o0], poses[o0 + 1], poses[o0 + 2]);
          s.dq ??= new THREE.Quaternion(poses[o0 + 4], poses[o0 + 5], poses[o0 + 6], poses[o0 + 3]);
          const tp = near ? fp : tmpP.set(poses[o0], poses[o0 + 1], poses[o0 + 2]);
          const tq = near ? r.bodies[s.fb].quaternion : tmpQ.set(poses[o0 + 4], poses[o0 + 5], poses[o0 + 6], poses[o0 + 3]);
          const k = 1 - Math.exp(-(near ? 14 : 5) * frameDt);
          s.disp.lerp(tp, k); s.dq.slerp(tq, k);
          fp.copy(s.disp); r.bodies[s.fb].quaternion.copy(s.dq);
        } else { s.disp = null; s.dq = null; }
        s.line.visible = onStem;   // once picked the stem is broken
        if (s.line.visible) { const arr = s.line.geometry.attributes.position; arr.setXYZ(1, fp.x, fp.y, fp.z + .04); arr.needsUpdate = true; }
      }
      r.harvested = rmeta.events.filter(e => e.event === 'harvest' && e.outcome === 'in_basket' && e.t <= lt).length;
      r.phase = (() => { const ev = [...rmeta.events].reverse().find(e => e.t <= lt && e.event !== 'harvest' && e.event !== 'retry'); return ev ? (ev.event === 'walk' ? 'walking to kiwi ' + (ev.fruit + 1) : 'harvesting kiwi ' + (ev.fruit + 1)) : 'walking'; })();
    }
  }

  // ---------- controls ----------
  const start = $('start');
  $('loading').textContent = 'ready';
  start.addEventListener('click', () => renderer.domElement.requestPointerLock());
  if (location.search.includes('nolock')) { start.style.display = 'none'; state.playing = true; }
  document.addEventListener('pointerlockchange', () => { const on = document.pointerLockElement === renderer.domElement; start.style.display = on ? 'none' : 'flex'; state.playing = true; });
  document.addEventListener('mousemove', e => { if (document.pointerLockElement !== renderer.domElement) return; player.yaw -= e.movementX * .0022; player.pitch = Math.max(-1.35, Math.min(1.35, player.pitch - e.movementY * .0022)); });
  addEventListener('keydown', e => { keys[e.code] = true; if (e.code === 'KeyF') toggleFollow(); if (e.code === 'KeyT') state.speed = state.speed === 1.5 ? 4 : 1.5; if (e.code === 'KeyH') $('hud').hidden = !$('hud').hidden; });
  addEventListener('keyup', e => keys[e.code] = false);
  addEventListener('resize', () => { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });
  function toggleFollow() {
    if (state.follow >= 0) { state.follow = -1; return; }
    let best = -1, bd = 1e9; state.robots.forEach((r, i) => { const d = r.world.distanceTo(player.pos); if (d < bd) { bd = d; best = i; } });
    state.follow = best;
  }
  const fwd = new THREE.Vector3(), right = new THREE.Vector3();
  function movePlayer(dt) {
    fwd.set(Math.cos(player.yaw), Math.sin(player.yaw), 0); right.set(Math.sin(player.yaw), -Math.cos(player.yaw), 0);
    const sp = (keys.ShiftLeft || keys.ShiftRight ? 4.2 : 2.0);
    const want = new THREE.Vector3();
    if (keys.KeyW || keys.ArrowUp) want.add(fwd); if (keys.KeyS || keys.ArrowDown) want.sub(fwd);
    if (keys.KeyD || keys.ArrowRight) want.add(right); if (keys.KeyA || keys.ArrowLeft) want.sub(right);
    if (want.lengthSq() > 0) want.normalize().multiplyScalar(sp);
    player.vel.lerp(want, 1 - Math.exp(-10 * dt));
    player.pos.addScaledVector(player.vel, dt);
    const lim = terrain.half - 1;
    player.pos.x = Math.max(-lim, Math.min(lim, player.pos.x)); player.pos.y = Math.max(-lim, Math.min(lim, player.pos.y));
    for (const p of orch.posts) { const dx = player.pos.x - p.x, dy = player.pos.y - p.y, d = Math.hypot(dx, dy); if (d < p.r && d > 1e-4) { player.pos.x = p.x + dx / d * p.r; player.pos.y = p.y + dy / d * p.r; } }
    for (const r of state.robots) { const c = r.bodies[meta.chassis].getWorldPosition(new THREE.Vector3()); const dx = player.pos.x - c.x, dy = player.pos.y - c.y, d = Math.hypot(dx, dy); if (d < .9 && d > 1e-4) { player.pos.x = c.x + dx / d * .9; player.pos.y = c.y + dy / d * .9; } }
    const bob = player.vel.length() > .3 ? Math.sin(performance.now() * .011) * .02 : 0;
    player.pos.z = terrain.sample(player.pos.x, player.pos.y) + player.eye + bob;
    camera.position.copy(player.pos);
    camera.lookAt(player.pos.x + Math.cos(player.yaw) * Math.cos(player.pitch), player.pos.y + Math.sin(player.yaw) * Math.cos(player.pitch), player.pos.z + Math.sin(player.pitch));
  }
  const followPos = new THREE.Vector3();
  function followCam(dt) {
    const r = state.robots[state.follow];
    const c = r.bodies[meta.chassis].getWorldPosition(new THREE.Vector3());
    const ang = player.yaw;
    followPos.set(c.x - Math.cos(ang) * 2.6, c.y - Math.sin(ang) * 2.6, c.z + 1.1);
    camera.position.lerp(followPos, 1 - Math.exp(-4 * dt));
    camera.lookAt(c.x, c.y, c.z + .5);
    player.pos.copy(camera.position); player.pos.z -= player.eye;
  }
  function labelNearest() {
    let best = null, bd = 1e9;
    for (const r of state.robots) { const c = r.bodies[meta.chassis].getWorldPosition(new THREE.Vector3()); const d = c.distanceTo(camera.position); if (d < bd) { bd = d; best = r; } }
    const lab = $('label');
    if (best && bd < 7) { lab.style.display = 'block'; lab.innerHTML = `<b>Spot ${best.id}</b> · ${best.phase} · <b>${best.harvested}</b>/${meta.fruits} in basket`; } else lab.style.display = 'none';
  }

  let last = performance.now();
  function frame(now) {
    const dt = Math.min(.05, (now - last) / 1000); last = now;
    if (state.playing) state.t += dt * state.speed;
    frameDt = dt * state.speed;
    poseRobots(state.t);
    if (state.follow >= 0) followCam(dt); else movePlayer(dt);
    // sun follows the player so the shadow frustum stays sharp
    sun.position.set(camera.position.x - 8, camera.position.y - 13, camera.position.z + 30);
    sun.target.position.set(camera.position.x, camera.position.y, 0);
    sun.shadow.camera.left = sun.shadow.camera.bottom = -22; sun.shadow.camera.right = sun.shadow.camera.top = 22; sun.shadow.camera.updateProjectionMatrix();
    $('count').textContent = state.robots.reduce((s, r) => s + (r.harvested || 0), 0);
    $('clock').textContent = `sim ${(state.t % dur).toFixed(0)} s · ×${state.speed}${state.follow >= 0 ? ' · following Spot ' + state.robots[state.follow].id : ''}`;
    labelNearest();
    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  }
  // put the player at the field edge looking in
  player.pos.set(2, -14, 0); player.yaw = Math.PI / 2;
  requestAnimationFrame(frame);
}
init().catch(e => { $('loading').textContent = 'failed to load: ' + e.message; console.error(e); });
