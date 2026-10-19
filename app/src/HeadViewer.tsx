import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

type Props = { url: string | null; shading: "texture" | "clay" };

type Loaded = { model: THREE.Object3D; textured: Map<THREE.Mesh, THREE.Material | THREE.Material[]> };

// Heads stay loaded for switching back and forth between versions; each is tens of MB.
const KEEP_LOADED = 4;

function dispose({ model, textured }: Loaded) {
  model.traverse((child) => {
    if (child instanceof THREE.Mesh) child.geometry.dispose();
  });
  for (const material of textured.values()) {
    for (const m of Array.isArray(material) ? material : [material]) {
      for (const value of Object.values(m)) if (value instanceof THREE.Texture) value.dispose();
      m.dispose();
    }
  }
}

// One reusable viewer: loads a GLB, frames it, and orbits. Later it can show bodies,
// props and sets too, so it knows nothing about heads beyond the default camera angle.
export function HeadViewer({ url, shading }: Props) {
  const mount = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<{
    scene: THREE.Scene;
    shown: Loaded | null;
    cache: Map<string, Loaded>;
    frame: (object: THREE.Object3D) => void;
  } | null>(null);
  const [status, setStatus] = useState<"empty" | "loading" | "ready" | "error">("empty");
  const [shownCount, setShownCount] = useState(0);

  useEffect(() => {
    const container = mount.current!;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    // Size the canvas by CSS: left to itself it grows by the display scale (125%, 150% in
    // Windows) and covers the version strip below, swallowing its clicks.
    Object.assign(renderer.domElement.style, { display: "block", width: "100%", height: "100%" });
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(30, 1, 0.01, 100);
    camera.position.set(0.6, 0.15, 1.6);
    scene.add(new THREE.HemisphereLight(0xffffff, 0x404040, 1.6));
    const key = new THREE.DirectionalLight(0xffffff, 1.8);
    key.position.set(1, 1.2, 2);
    scene.add(key);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    // Points the camera at a newly loaded model from a three-quarter front angle.
    const frameModel = (object: THREE.Object3D) => {
      const box = new THREE.Box3().setFromObject(object);
      const size = box.getSize(new THREE.Vector3()).length();
      const center = box.getCenter(new THREE.Vector3());
      controls.target.copy(center);
      camera.position.copy(center).add(new THREE.Vector3(0.35, 0.1, 1).normalize().multiplyScalar(size * 2.2));
      camera.near = size / 100;
      camera.far = size * 100;
      camera.updateProjectionMatrix();
    };
    const cache = new Map<string, Loaded>();
    sceneRef.current = { scene, shown: null, cache, frame: frameModel };

    const resize = () => {
      const { clientWidth: w, clientHeight: h } = container;
      renderer.setSize(w, h, false);
      camera.aspect = w / Math.max(h, 1);
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(container);
    resize();

    let frame = 0;
    const loop = () => {
      controls.update();
      renderer.render(scene, camera);
      frame = requestAnimationFrame(loop);
    };
    loop();

    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      controls.dispose();
      for (const loaded of cache.values()) dispose(loaded);
      renderer.dispose();
      container.removeChild(renderer.domElement);
      sceneRef.current = null;
    };
  }, []);

  useEffect(() => {
    const state = sceneRef.current;
    if (!state) return;
    if (state.shown) {
      state.scene.remove(state.shown.model);
      state.shown = null;
    }
    if (!url) {
      setStatus("empty");
      return;
    }
    const show = (loaded: Loaded) => {
      // Most recently shown last, so the oldest is first to go.
      state.cache.delete(url);
      state.cache.set(url, loaded);
      for (const [key, old] of state.cache) {
        if (state.cache.size <= KEEP_LOADED) break;
        state.cache.delete(key);
        dispose(old);
      }
      state.shown = loaded;
      state.scene.add(loaded.model);
      state.frame(loaded.model);
      setStatus("ready");
      setShownCount((n) => n + 1);
    };
    const cached = state.cache.get(url);
    if (cached) {
      show(cached);
      return;
    }
    let cancelled = false;
    setStatus("loading");
    new GLTFLoader().load(
      url,
      (gltf) => {
        const loaded: Loaded = { model: gltf.scene, textured: new Map() };
        gltf.scene.traverse((child) => {
          if (child instanceof THREE.Mesh) loaded.textured.set(child, child.material);
        });
        if (!sceneRef.current) return dispose(loaded);
        if (cancelled) {
          // Another version was picked meanwhile; keep this one for when it's picked again.
          if (!state.cache.has(url)) state.cache.set(url, loaded);
          return;
        }
        show(loaded);
      },
      undefined,
      () => !cancelled && setStatus("error"),
    );
    return () => {
      cancelled = true;
    };
  }, [url]);

  useEffect(() => {
    const shown = sceneRef.current?.shown;
    if (!shown) return;
    const clay = new THREE.MeshStandardMaterial({ color: 0xb8b2ac, roughness: 0.6 });
    for (const [mesh, material] of shown.textured) mesh.material = shading === "clay" ? clay : material;
    return () => clay.dispose();
  }, [shading, shownCount]);

  return (
    <div className="relative h-full w-full">
      <div ref={mount} className="absolute inset-0 overflow-hidden" />
      {status !== "ready" && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center text-sm text-neutral-500">
          {status === "empty" && "No head to show yet"}
          {status === "loading" && "Loading the head…"}
          {status === "error" && "This run has no viewable model"}
        </div>
      )}
    </div>
  );
}
