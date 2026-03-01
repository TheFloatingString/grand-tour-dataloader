"use client";

import dynamic from "next/dynamic";

const CanvasContent = dynamic(
  () => import("./CanvasContent").then((mod) => mod.CanvasContent),
  { ssr: false }
);

export function URDFViewer() {
  return (
    <div style={{ width: "100%", height: "100%" }}>
      <CanvasContent />
    </div>
  );
}
