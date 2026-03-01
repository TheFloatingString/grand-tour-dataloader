"use client";

import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useEffect, useRef, useState } from "react";
import {
  BufferAttribute,
  BufferGeometry,
  DoubleSide,
  Euler,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  Object3D,
  Quaternion,
  Vector3,
} from "three";
import URDFLoader from "urdf-loader";

const robotMaterial = new MeshStandardMaterial({ color: 0x4a90d9 });

type JointMap = Record<string, any>;

interface PlaybackData {
  joint_names: string[];
  fps: number;
  duration: number;
  timestamps: number[];
  frames: number[][];
}

interface BasePosData {
  fps: number;
  duration: number;
  timestamps: number[];
  frames: number[][];  // Nx7: [x, y, z, qx, qy, qz, qw]
}

interface PlaybackState {
  data: PlaybackData | null;
  joints: JointMap;
  playing: boolean;
  frameIdx: number;
  elapsed: number;
}

function RobotLoader({
  onJoints,
  robotRef,
}: {
  onJoints: (j: JointMap) => void;
  robotRef: React.MutableRefObject<Object3D | null>;
}) {
  const { scene } = useThree();

  useEffect(() => {
    const loader = new URDFLoader();
    loader.packages = {
      anymal_d_simple_description: "/anymal_d_simple_description",
    };
    loader.load(
      "/anymal_d_simple_description/urdf/anymal.urdf",
      (robot) => {
        robot.traverse((child: any) => {
          if (child.isMesh) {
            child.visible = true;
            child.material = robotMaterial;
            child.renderOrder = 1;
          }
        });
        robot.position.y = 0.5;
        scene.add(robot);
        robotRef.current = robot;
        onJoints((robot as any).joints ?? {});
      },
      undefined,
      (err) => console.error("URDF load error:", err)
    );
  }, [scene, onJoints, robotRef]);

  return null;
}

// Base rotation applied to robot mesh at load time (ROS Z-up -> Three.js Y-up)
const BASE_ROT = new Quaternion().setFromEuler(new Euler(-Math.PI / 2, 0, 0));

function BasePosDriver({
  basePosData,
  stateRef,
  robotRef,
  orbitRef,
}: {
  basePosData: BasePosData;
  stateRef: React.MutableRefObject<PlaybackState>;
  robotRef: React.MutableRefObject<Object3D | null>;
  orbitRef: React.MutableRefObject<any>;
}) {
  const initialOdomRef = useRef<Vector3 | null>(null);
  const prevRobotPosRef = useRef<Vector3 | null>(null);

  useFrame(() => {
    if (!robotRef.current || !orbitRef.current) return;
    const idx = Math.min(
      Math.floor(stateRef.current.elapsed * basePosData.fps),
      basePosData.frames.length - 1
    );
    const f = basePosData.frames[idx];
    if (!f) return;

    if (!initialOdomRef.current) {
      initialOdomRef.current = new Vector3(f[0], f[1], f[2]);
    }

    // ROS odom: x=forward, y=left, z=up -> Three.js: x, z=-y, y=z
    const rel = new Vector3(
      f[0] - initialOdomRef.current.x,
      f[2] - initialOdomRef.current.z + 0.5,
      -(f[1] - initialOdomRef.current.y)
    );

    if (prevRobotPosRef.current) {
      const delta = rel.clone().sub(prevRobotPosRef.current);
      orbitRef.current.target.add(delta);
      orbitRef.current.object.position.add(delta);
    }

    robotRef.current.position.set(rel.x, rel.y, rel.z);
    prevRobotPosRef.current = rel.clone();

    // Apply orientation: compose base axis-swap with ROS quaternion [qx, qy, qz, qw]
    if (f.length >= 7) {
      const rosQ = new Quaternion(f[3], f[4], f[5], f[6]);
      robotRef.current.quaternion.copy(BASE_ROT).multiply(rosQ);
    }
  });

  return null;
}

function viridis(t: number): [number, number, number] {
  const stops: [number, number, number][] = [
    [0.267, 0.004, 0.329],
    [0.231, 0.322, 0.545],
    [0.129, 0.569, 0.549],
    [0.369, 0.788, 0.384],
    [0.993, 0.906, 0.145],
  ];
  const s = Math.min(Math.max(t, 0), 1) * (stops.length - 1);
  const i = Math.floor(s);
  const f = s - i;
  const a = stops[Math.min(i, stops.length - 1)];
  const b = stops[Math.min(i + 1, stops.length - 1)];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

const MAX_RIBBON = 5000;
const RIBBON_HALF_W = 0.4;
const WORLD_UP = new Vector3(0, 1, 0);

function MotionTrail({ robotRef }: { robotRef: React.MutableRefObject<Object3D | null> }) {
  const { scene } = useThree();
  const meshRef = useRef<Mesh | null>(null);
  const posArr = useRef(new Float32Array(MAX_RIBBON * 2 * 3));
  const colArr = useRef(new Float32Array(MAX_RIBBON * 2 * 3));
  const ctrPts = useRef<Vector3[]>([]);
  const heights = useRef<number[]>([]);
  const lastRef = useRef<Vector3 | null>(null);

  useEffect(() => {
    const geo = new BufferGeometry();

    // Fixed quad-strip index buffer: 2 tris per quad, (MAX_RIBBON-1) quads
    const idx = new Uint32Array((MAX_RIBBON - 1) * 6);
    for (let i = 0; i < MAX_RIBBON - 1; i++) {
      idx[i * 6 + 0] = 2 * i;     idx[i * 6 + 1] = 2 * i + 1; idx[i * 6 + 2] = 2 * i + 2;
      idx[i * 6 + 3] = 2 * i + 1; idx[i * 6 + 4] = 2 * i + 3; idx[i * 6 + 5] = 2 * i + 2;
    }
    geo.setAttribute("position", new BufferAttribute(posArr.current, 3));
    geo.setAttribute("color",    new BufferAttribute(colArr.current, 3));
    geo.setIndex(new BufferAttribute(idx, 1));
    geo.setDrawRange(0, 0);

    const mat = new MeshBasicMaterial({ vertexColors: true, side: DoubleSide, depthTest: false, depthWrite: false });
    const mesh = new Mesh(geo, mat);
    mesh.renderOrder = 0;
    meshRef.current = mesh;
    scene.add(mesh);
    return () => { scene.remove(mesh); mat.dispose(); geo.dispose(); };
  }, [scene]);

  useFrame(() => {
    if (!robotRef.current || !meshRef.current) return;

    const wp = new Vector3();
    robotRef.current.getWorldPosition(wp);
    const height = wp.y;
    const gp = new Vector3(wp.x, wp.y - 0.6, wp.z);

    if (lastRef.current && lastRef.current.distanceTo(gp) > 5) {
      ctrPts.current = []; heights.current = []; lastRef.current = null;
      meshRef.current.geometry.setDrawRange(0, 0);
      return;
    }
    if (lastRef.current && lastRef.current.distanceTo(gp) < 0.1) return;

    if (ctrPts.current.length >= MAX_RIBBON) {
      ctrPts.current = ctrPts.current.slice(-MAX_RIBBON + 1);
      heights.current = heights.current.slice(-MAX_RIBBON + 1);
    }
    ctrPts.current.push(gp.clone());
    heights.current.push(height);
    lastRef.current = gp.clone();

    const n = ctrPts.current.length;
    if (n < 2) return;

    let minH = heights.current[0], maxH = heights.current[0];
    for (const h of heights.current) { if (h < minH) minH = h; if (h > maxH) maxH = h; }
    const range = maxH - minH || 1;

    const dir = new Vector3(), right = new Vector3();
    for (let i = 0; i < n; i++) {
      const pt = ctrPts.current[i];
      dir.subVectors(i < n - 1 ? ctrPts.current[i + 1] : pt, i > 0 ? ctrPts.current[i - 1] : pt);
      dir.y = 0; dir.normalize();
      right.crossVectors(dir, WORLD_UP).normalize();

      const vi = i * 2;
      posArr.current.set([pt.x - right.x * RIBBON_HALF_W, pt.y, pt.z - right.z * RIBBON_HALF_W], vi * 3);
      posArr.current.set([pt.x + right.x * RIBBON_HALF_W, pt.y, pt.z + right.z * RIBBON_HALF_W], (vi + 1) * 3);

      const [r, g, b] = viridis((heights.current[i] - minH) / range);
      colArr.current.set([r, g, b], vi * 3);
      colArr.current.set([r, g, b], (vi + 1) * 3);
    }

    const geo = meshRef.current.geometry;
    (geo.attributes.position as BufferAttribute).needsUpdate = true;
    (geo.attributes.color as BufferAttribute).needsUpdate = true;
    geo.setDrawRange(0, (n - 1) * 6);
    geo.computeBoundingSphere();
  });

  return null;
}

function PlaybackDriver({
  stateRef,
  onFrame,
}: {
  stateRef: React.MutableRefObject<PlaybackState>;
  onFrame: (idx: number, elapsed: number) => void;
}) {
  useFrame((_, delta) => {
    const s = stateRef.current;
    if (!s.playing || !s.data) return;

    s.elapsed += delta;
    if (s.elapsed > s.data.duration) s.elapsed = 0;

    const idx = Math.min(
      Math.floor(s.elapsed * s.data.fps),
      s.data.frames.length - 1
    );

    if (idx !== s.frameIdx) {
      s.frameIdx = idx;
      const frame = s.data.frames[idx];
      s.data.joint_names.forEach((name, i) => {
        s.joints[name]?.setJointValue(frame[i]);
      });
      onFrame(idx, s.elapsed);
    }
  });

  return null;
}

const N_SECONDS = 5;

function Sparkline({ values }: { values: number[] }) {
  if (values.length < 2) return <div className="h-8 w-full rounded bg-zinc-100 dark:bg-zinc-800" />;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const W = 200;
  const H = 32;
  const pts = values
    .map((v, i) => `${(i / (values.length - 1)) * W},${H - ((v - min) / range) * H}`)
    .join(" ");
  return (
    <svg width="100%" height={H} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
      <polyline points={pts} fill="none" stroke="#6366f1" strokeWidth="1.5" strokeLinejoin="round" />
    </svg>
  );
}

export function CanvasContent() {
  const [joints, setJoints] = useState<JointMap>({});
  const [playbackData, setPlaybackData] = useState<PlaybackData | null>(null);
  const [playing, setPlaying] = useState(false);
  const [frameIdx, setFrameIdx] = useState(0);
  const [autoRotate, setAutoRotate] = useState(true);

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "r" || e.key === "R") setAutoRotate((v) => !v);
      if (e.key === "p" || e.key === "P") setPlaying((v) => !v);
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, []);

  const [basePosData, setBasePosData] = useState<BasePosData | null>(null);
  const basePosHistoryRef = useRef<{ t: number; xyz: number[] }[]>([]);

  const robotRef = useRef<Object3D | null>(null);
  const historyRef = useRef<{ t: number; vals: number[] }[]>([]);
  const orbitRef = useRef<any>(null);

  useEffect(() => {
    if (orbitRef.current) {
      orbitRef.current.autoRotate = playing && autoRotate;
    }
  }, [playing, autoRotate]);

  const stateRef = useRef<PlaybackState>({
    data: null,
    joints: {},
    playing: false,
    frameIdx: 0,
    elapsed: 0,
  });

  const onJoints = useRef((j: JointMap) => {
    setJoints(j);
    stateRef.current.joints = j;
  });

  useEffect(() => {
    fetch("/joint_data.json")
      .then((r) => r.json())
      .then((data: PlaybackData) => {
        setPlaybackData(data);
        stateRef.current.data = data;
      })
      .catch(() => {});
    fetch("/base_position.json")
      .then((r) => r.json())
      .then((data: BasePosData) => setBasePosData(data))
      .catch(() => {});
  }, []);

  useEffect(() => {
    stateRef.current.playing = playing;
  }, [playing]);

  const movableJoints = Object.entries(joints).filter(
    ([, j]) => j.jointType !== "fixed"
  );

  const duration = playbackData?.duration ?? 0;
  const currentTime = playbackData?.timestamps[frameIdx] ?? 0;

  const handleScrub = (e: React.ChangeEvent<HTMLInputElement>) => {
    const t = parseFloat(e.target.value);
    if (!playbackData) return;
    const idx = Math.min(
      Math.floor(t * playbackData.fps),
      playbackData.frames.length - 1
    );
    stateRef.current.elapsed = t;
    stateRef.current.frameIdx = idx;
    setFrameIdx(idx);
    const frame = playbackData.frames[idx];
    playbackData.joint_names.forEach((name, i) => {
      stateRef.current.joints[name]?.setJointValue(frame[i]);
    });
  };

  return (
    <div className="flex flex-col w-full h-full">
      <div className="flex flex-1 overflow-hidden">
        <Canvas
          style={{ flex: 1, height: "100%", background: "#e9e6ff" }}
          camera={{ position: [1.5, 1.5, 1.5], fov: 50 }}
        >
          <OrbitControls ref={orbitRef} autoRotateSpeed={2} />
          <ambientLight intensity={0.6} />
          <directionalLight position={[5, 10, 5]} intensity={1} />
          <gridHelper args={[500, 500, "#cccccc", "#e0e0e0"]} />
          <axesHelper args={[0.5]} />
          <MotionTrail robotRef={robotRef} />
          <RobotLoader onJoints={onJoints.current} robotRef={robotRef} />
          {basePosData && (
            <BasePosDriver
              basePosData={basePosData}
              stateRef={stateRef}
              robotRef={robotRef}
              orbitRef={orbitRef}
            />
          )}
          <PlaybackDriver
            stateRef={stateRef}
            onFrame={(idx, elapsed) => {
              const frame = stateRef.current.data?.frames[idx];
              if (frame) {
                historyRef.current.push({ t: elapsed, vals: frame });
                historyRef.current = historyRef.current.filter(
                  (e) => elapsed - e.t <= N_SECONDS
                );
              }
              if (basePosData) {
                const bIdx = Math.min(
                  Math.floor(elapsed * basePosData.fps),
                  basePosData.frames.length - 1
                );
                const bFrame = basePosData.frames[bIdx];
                if (bFrame) {
                  basePosHistoryRef.current.push({ t: elapsed, xyz: bFrame });
                  basePosHistoryRef.current = basePosHistoryRef.current.filter(
                    (e) => elapsed - e.t <= N_SECONDS
                  );
                }
              }
              setFrameIdx(idx);
            }}
          />
        </Canvas>
        <div className="sidebar-scroll w-64 h-full overflow-y-auto bg-white dark:bg-zinc-900 border-l p-3 flex flex-col gap-3">
          <div className="flex flex-col gap-1">
            <h2 className="font-semibold text-sm">Mission</h2>
            <p className="font-mono text-xs text-zinc-500">2024-10-01-11-29-55</p>
          </div>
          <hr className="border-zinc-200 dark:border-zinc-700" />
          <div className="flex flex-col gap-1">
            <h2 className="font-semibold text-sm mb-1">Keybindings</h2>
            <div className="flex items-center justify-between text-xs text-zinc-500">
              <kbd className="font-mono bg-zinc-100 dark:bg-zinc-800 px-1 rounded">P</kbd>
              <span>{playing ? "Pause" : "Play"}</span>
            </div>
            <div className="flex items-center justify-between text-xs text-zinc-500">
              <kbd className="font-mono bg-zinc-100 dark:bg-zinc-800 px-1 rounded">R</kbd>
              <span>Auto-rotate {autoRotate ? "(on)" : "(off)"}</span>
            </div>
          </div>
          <hr className="border-zinc-200 dark:border-zinc-700" />
          <h2 className="font-semibold text-sm">Base Position</h2>
          {basePosData ? (
            ["x", "y", "z"].map((axis, i) => {
              const bIdx = Math.min(
                Math.floor((playbackData?.timestamps[frameIdx] ?? 0) * basePosData.fps),
                basePosData.frames.length - 1
              );
              const val = basePosData.frames[bIdx]?.[i] ?? 0;
              const sparkVals = basePosHistoryRef.current.map((e) => e.xyz[i] ?? 0);
              return (
                <div key={axis} className="flex flex-col gap-0.5">
                  <div className="flex justify-between text-xs font-mono">
                    <span className="text-zinc-500">{axis}</span>
                    <span className="text-zinc-800 dark:text-zinc-200">{val.toFixed(3)} m</span>
                  </div>
                  <Sparkline values={sparkVals} />
                </div>
              );
            })
          ) : (
            <p className="text-xs text-zinc-400">No data</p>
          )}
          <hr className="border-zinc-200 dark:border-zinc-700" />
          <h2 className="font-semibold text-sm">Orientation (RPY °)</h2>
          {basePosData ? (() => {
            const bIdx = Math.min(
              Math.floor((playbackData?.timestamps[frameIdx] ?? 0) * basePosData.fps),
              basePosData.frames.length - 1
            );
            const f = basePosData.frames[bIdx];
            const toRPY = (frame: number[]) => {
              if (frame.length < 7) return [0, 0, 0];
              const eu = new Euler().setFromQuaternion(
                new Quaternion(frame[3], frame[4], frame[5], frame[6]),
                "ZYX"
              );
              // ZYX: eu.z=yaw, eu.y=pitch, eu.x=roll
              return [eu.x, eu.y, eu.z].map((v) => (v * 180) / Math.PI);
            };
            const rpyDeg = f ? toRPY(f) : [0, 0, 0];
            return (["roll", "pitch", "yaw"] as const).map((label, i) => {
              const sparkVals = basePosHistoryRef.current.map((e) => toRPY(e.xyz)[i]);
              return (
                <div key={label} className="flex flex-col gap-0.5">
                  <div className="flex justify-between text-xs font-mono">
                    <span className="text-zinc-500">{label}</span>
                    <span className="text-zinc-800 dark:text-zinc-200">{rpyDeg[i].toFixed(1)}°</span>
                  </div>
                  <Sparkline values={sparkVals} />
                </div>
              );
            });
          })() : (
            <p className="text-xs text-zinc-400">No data</p>
          )}
          <hr className="border-zinc-200 dark:border-zinc-700" />
          <h2 className="font-semibold text-sm">Joints</h2>
          {movableJoints.length === 0 && (
            <p className="text-xs text-zinc-400">Loading...</p>
          )}
          {movableJoints.map(([name]) => {
            const nameIdx = playbackData?.joint_names.indexOf(name) ?? -1;
            const val = nameIdx >= 0 ? (playbackData!.frames[frameIdx]?.[nameIdx] ?? 0) : 0;
            const sparkVals = nameIdx >= 0
              ? historyRef.current.map((e) => e.vals[nameIdx] ?? 0)
              : [];
            return (
              <div key={name} className="flex flex-col gap-0.5">
                <div className="flex justify-between text-xs font-mono">
                  <span className="truncate max-w-[140px] text-zinc-500">{name}</span>
                  <span className="text-zinc-800 dark:text-zinc-200">{val.toFixed(3)}</span>
                </div>
                <Sparkline values={sparkVals} />
              </div>
            );
          })}
        </div>
      </div>

      {playbackData && (
        <div className="flex items-center gap-3 px-4 py-2 bg-white dark:bg-zinc-900 border-t text-sm">
          <button
            onClick={() => setPlaying((p) => !p)}
            className="w-16 rounded bg-zinc-800 text-white px-2 py-1 text-xs hover:bg-zinc-700"
          >
            {playing ? "Pause" : "Play"}
          </button>
          <input
            type="range"
            min={0}
            max={duration}
            step={1 / playbackData.fps}
            value={currentTime}
            onChange={handleScrub}
            className="flex-1"
          />
          <span className="font-mono text-xs text-zinc-500 w-24 text-right">
            {currentTime.toFixed(1)}s / {duration.toFixed(1)}s
          </span>
        </div>
      )}
    </div>
  );
}
