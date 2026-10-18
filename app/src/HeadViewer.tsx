import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

type Props = { url: string | null; shading: "texture" | "clay" };

// One reusable viewer: loads a GLB, frames it, and orbits. Later it can show bodies,
// props and sets too, so it knows nothing about heads beyond the default camera angle.
export function HeadViewer({ url, shading }: Props) {
  const mount = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<{
    scene: THREE.Scene;
    model: THREE.Object3D | null;
    textured: Map<THREE.Mesh, THREE.Material | THREE.Material[]>;
    frame: (object: THREE.Object3D) => void;
  } | null>(null);
  const [status, setStatus] = useState<"empty" | "loading" | "ready" | "error">("empty");

  useEffect(() => {
    const container = mount.current!;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
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
    sceneRef.current = { scene, model: null, textured: new Map(), frame: frameModel };

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
      renderer.dispose();
      container.removeChild(renderer.domElement);
      sceneRef.current = null;
    };
  }, []);

  useEffect(() => {
    const state = sceneRef.current;
    if (!state) return;
    if (state.model) {
      state.scene.remove(state.model);
      state.model = null;
      state.textured.clear();
    }
    if (!url) {
      setStatus("empty");
      return;
    }
    let cancelled = false;
    setStatus("loading");
    new GLTFLoader().load(
      url,
      (gltf) => {
        if (cancelled || !sceneRef.current) return;
        const model = gltf.scene;
        model.traverse((child) => {
          if (child instanceof THREE.Mesh) state.textured.set(child, child.material);
        });
        state.model = model;
        state.scene.add(model);
        state.frame(model);
        setStatus("ready");
      },
      undefined,
      () => !cancelled && setStatus("error"),
    );
    return () => {
      cancelled = true;
    };
  }, [url]);

  useEffect(() => {
    const state = sceneRef.current;
    if (!state || status !== "ready") return;
    const clay = new THREE.MeshStandardMaterial({ color: 0xb8b2ac, roughness: 0.6 });
    for (const [mesh, material] of state.textured) mesh.material = shading === "clay" ? clay : material;
    return () => clay.dispose();
  }, [shading, status]);

  return (
    <div className="relative h-full w-full">
      <div ref={mount} className="absolute inset-0" />
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
