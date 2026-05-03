#!/usr/bin/env bun
/**
 * render-architecture.ts — Generate the go-pherence architecture SVG diagram.
 *
 * Usage: bun run scripts/render-architecture.ts [--output docs/architecture.svg]
 *
 * Uses the taoofmac.com portfolio diagram style:
 *   - Dark/light mode via prefers-color-scheme
 *   - Box classes: box, box-blue, box-green, box-amber, box-purple, box-red, box-gpu
 *   - Monospace perf annotations
 *   - Arrow markers with directional heads
 *
 * @module render-architecture
 * @kind entrypoint
 */

import { writeFileSync } from "fs";
import { resolve } from "path";

// --- Config ---
interface Box {
  x: number; y: number; w: number; h: number;
  cls: string; label: string; sub?: string;
}
interface Zone {
  x: number; y: number; w: number; h: number;
  label: string; boxes: Box[];
}

const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const W = 900, PAD = 24;
const ZW = W - PAD * 2; // zone width

// --- Style (dark/light responsive) ---
const STYLE = `
    @media (prefers-color-scheme: dark) {
      .bg { fill: transparent; }
      .box { fill: #1a1e2a; stroke: #2a3040; stroke-width: 1.5; }
      .box-blue { fill: #0d1e38; stroke: #2b5cb0; stroke-width: 1.5; }
      .box-green { fill: #0d2220; stroke: #207060; stroke-width: 1.5; }
      .box-amber { fill: #221a10; stroke: #a06020; stroke-width: 1.5; }
      .box-purple { fill: #1a0d28; stroke: #7030a0; stroke-width: 1.5; }
      .box-red { fill: #280d0d; stroke: #a03030; stroke-width: 1.5; }
      .box-gpu { fill: #0d2810; stroke: #30a040; stroke-width: 2; }
      .label { fill: #d0daf0; } .sub { fill: #5070a0; } .perf { fill: #70a050; }
      .arrow { stroke: #5070a0; } .arrow-gpu { stroke: #30a040; }
      .title { fill: #e0e8f8; } .zone { fill: #0a0e16; stroke: #1a2030; }
      .zone-label { fill: #3a4a6a; }
    }
    @media (prefers-color-scheme: light) {
      .bg { fill: transparent; }
      .box { fill: #ffffff; stroke: #c8d0e0; stroke-width: 1.5; }
      .box-blue { fill: #dbeafe; stroke: #3b82f6; stroke-width: 1.5; }
      .box-green { fill: #d1fae5; stroke: #059669; stroke-width: 1.5; }
      .box-amber { fill: #fef3c7; stroke: #d97706; stroke-width: 1.5; }
      .box-purple { fill: #ede9fe; stroke: #7c3aed; stroke-width: 1.5; }
      .box-red { fill: #fee2e2; stroke: #dc2626; stroke-width: 1.5; }
      .box-gpu { fill: #dcfce7; stroke: #16a34a; stroke-width: 2; }
      .label { fill: #1a2a40; } .sub { fill: #5070a0; } .perf { fill: #15803d; }
      .arrow { stroke: #5070a0; } .arrow-gpu { stroke: #16a34a; }
      .title { fill: #0f172a; } .zone { fill: #f1f5f9; stroke: #cbd5e1; }
      .zone-label { fill: #94a3b8; }
    }
    text { font-family: -apple-system, "Segoe UI", Helvetica, sans-serif; }
    .label { font-size: 12px; font-weight: 600; }
    .sub { font-size: 10px; }
    .perf { font-size: 9px; font-family: "SF Mono", Menlo, monospace; }
    .title { font-size: 16px; font-weight: 700; }
    .zone-label { font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
`;

// --- Rendering helpers ---
function renderBox(b: Box): string {
  const cx = b.x + b.w / 2;
  let s = `    <rect x="${b.x}" y="${b.y}" width="${b.w}" height="${b.h}" rx="5" class="${b.cls}"/>\n`;
  s += `    <text x="${cx}" y="${b.y + (b.sub ? 15 : b.h / 2 + 4)}" class="label" text-anchor="middle">${esc(b.label)}</text>\n`;
  if (b.sub) s += `    <text x="${cx}" y="${b.y + 28}" class="perf" text-anchor="middle">${esc(b.sub)}</text>\n`;
  return s;
}

// --- Architecture data ---
const layers = [
  {
    label: "User API", y: 60, h: 50,
    boxes: [
      { x: 12, w: 160, cls: "box-purple", label: "cmd/llmgen", sub: "-gpu flag" },
      { x: 184, w: 160, cls: "box-purple", label: "cmd/tinydemo" },
      { x: 356, w: 240, cls: "box", label: "LoadLlama() → Generate()" },
      { x: 608, w: 232, cls: "box-gpu", label: "LoadGPUModel() → Generate()" },
    ]
  },
  {
    label: "Models (model/)", y: 124, h: 110,
    boxes: [
      { x: 12, w: 130, cls: "box-blue", label: "BERT Encoder", sub: "GTE-small · 10.8ms" },
      { x: 154, w: 150, cls: "box-blue", label: "LLaMA Decoder", sub: "RoPE · GQA · KV Cache" },
      { x: 316, w: 120, cls: "box", label: "BPE Tokenizer", sub: "GPT-2 + Qwen" },
      { x: 448, w: 130, cls: "box", label: "NN Modules", sub: "Linear · LayerNorm · Emb" },
      { x: 590, w: 250, cls: "box-gpu", label: "GPU Forward Pass", sub: "DevBuf dispatch · VRAM weights" },
      { x: 12, w: 200, cls: "box-amber", label: "GPTQ INT4 Dequant", sub: undefined, yOff: 46 },
      { x: 224, w: 200, cls: "box", label: "Persistent Workspace", sub: undefined, yOff: 46 },
    ]
  },
  {
    label: "Tensor Framework (tensor/)", y: 248, h: 90, w: 420,
    boxes: [
      { x: 12, w: 120, cls: "box-blue", label: "UOp DAG" },
      { x: 144, w: 120, cls: "box-blue", label: "Fusion Engine" },
      { x: 276, w: 130, cls: "box-blue", label: "Pattern Rewrite" },
      { x: 12, w: 120, cls: "box", label: "Shape · DType", yOff: 36 },
      { x: 144, w: 120, cls: "box", label: "Realize · Pool", yOff: 36 },
      { x: 276, w: 130, cls: "box", label: "16 rewrite rules", yOff: 36 },
    ]
  },
  {
    label: "Safetensors (safetensors/)", y: 248, h: 90, x: 456, w: 420,
    boxes: [
      { x: 12, w: 120, cls: "box-amber", label: "F16 / BF16 / F32" },
      { x: 144, w: 120, cls: "box-amber", label: "Sharded Models" },
      { x: 276, w: 130, cls: "box-amber", label: "GPTQ INT4 Meta" },
      { x: 12, w: 394, cls: "box", label: "HuggingFace · mmap · GetFloat32/Int32/Raw", yOff: 36 },
    ]
  },
  {
    label: "GPU Compute (gpu/) — RTX 3060 · 28 SMs · 12GB VRAM", y: 352, h: 140,
    boxes: [
      { x: 12, w: 180, cls: "box-gpu", label: "DevBuf", sub: "CPU↔GPU · Lazy · MarkDirty" },
      { x: 204, w: 410, cls: "box-gpu", label: "PTX Kernels (8 total)", sub: "sgemm 348G · q4 197µs · add · mul · scale · silu · rmsnorm" },
      { x: 626, w: 214, cls: "box-green", label: "purego CUDA Driver", sub: "dlopen(libcuda.so.1) · 14 funcs" },
      { x: 12, w: 300, cls: "box-red", label: "NV Direct ioctl", sub: "/dev/nvidia* · RM · UUID · 84 SMs · Channel", yOff: 56 },
      { x: 324, w: 290, cls: "box-gpu", label: "GPU INT4 Weight Store", sub: "3.2GB VRAM for 7B · Fused dequant", yOff: 56 },
      { x: 626, w: 214, cls: "box", label: "CPU SIMD Fallback", sub: "AVX2+FMA (amd64) · NEON (arm64)", yOff: 56 },
    ]
  },
  {
    label: "Hardware", y: 506, h: 50,
    boxes: [
      { x: 12, w: 200, cls: "box-green", label: "RTX 3060 · 12GB VRAM" },
      { x: 224, w: 200, cls: "box", label: "6-core CPU · 64GB RAM" },
      { x: 436, w: 200, cls: "box", label: "Proxmox VM (borg)" },
      { x: 648, w: 192, cls: "box", label: "GPU Passthrough PCIe" },
    ]
  },
  {
    label: "Performance", y: 570, h: 60,
    boxes: [
      { x: 12, w: 160, cls: "box-blue", label: "SmolLM2-135M", sub: "28 tok/s · 35ms" },
      { x: 184, w: 160, cls: "box-amber", label: "Qwen2.5-7B INT4", sub: "1.0 tok/s · 1009ms" },
      { x: 356, w: 160, cls: "box-green", label: "GPU SGEMM", sub: "348 GFLOPS · 1024²" },
      { x: 528, w: 160, cls: "box-green", label: "GPU Q4 GEMV", sub: "197µs · 3584² · 1.7e-6" },
      { x: 700, w: 140, cls: "box-blue", label: "GTE-small", sub: "10.8ms · 0 allocs" },
    ]
  },
];

// --- Build SVG ---
const H = 640;
let svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" font-family="system-ui,-apple-system,sans-serif">\n`;
svg += `  <style>${STYLE}  </style>\n`;
svg += `  <defs>\n`;
svg += `    <marker id="ah" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8z" fill="#5070a0" stroke="none"/></marker>\n`;
svg += `    <marker id="ahg" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8z" fill="#16a34a" stroke="none"/></marker>\n`;
svg += `  </defs>\n`;

// Title
svg += `  <text x="${PAD}" y="28" class="title">go-pherence Architecture</text>\n`;
svg += `  <text x="${PAD}" y="44" class="sub">Pure Go + SIMD Assembly + PTX GPU · Zero CGo · Static Binary</text>\n`;

// Render layers
for (const layer of layers) {
  const lx = (layer as any).x ?? 0;
  const lw = (layer as any).w ?? ZW;
  svg += `  <g transform="translate(${PAD + lx},${layer.y})">\n`;
  svg += `    <rect x="0" y="0" width="${lw}" height="${layer.h}" rx="8" class="zone"/>\n`;
  svg += `    <text x="12" y="14" class="zone-label">${layer.label}</text>\n`;
  for (const box of layer.boxes) {
    const by = (box as any).yOff ? 22 + (box as any).yOff : 22;
    const bh = box.sub ? 36 : (layer.h > 60 ? 30 : 24);
    svg += renderBox({ ...box, y: by, h: bh });
  }
  svg += `  </g>\n`;
}

// Flow arrows
const arrows = [
  [234, 110, 234, 124],
  [500, 110, 500, 124],
  [234, 234, 234, 248],
  [666, 234, 666, 248],
];
for (const [x1, y1, x2, y2] of arrows) {
  svg += `  <line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" class="arrow" stroke-width="1.2" marker-end="url(#ah)"/>\n`;
}
svg += `  <line x1="450" y1="338" x2="450" y2="352" class="arrow-gpu" stroke-width="1.5" marker-end="url(#ahg)"/>\n`;

svg += `</svg>\n`;

// --- Output ---
const outPath = process.argv.includes("--output")
  ? process.argv[process.argv.indexOf("--output") + 1]
  : resolve(__dirname, "../docs/architecture.svg");

writeFileSync(outPath, svg);
console.log(`Wrote ${outPath} (${svg.length} bytes)`);
