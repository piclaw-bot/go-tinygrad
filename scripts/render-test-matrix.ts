#!/usr/bin/env bun
/**
 * render-test-matrix.ts — Generate the go-tinygrad test matrix SVG.
 *
 * Usage: bun run scripts/render-test-matrix.ts [--output docs/test-matrix.svg]
 *
 * Palette semantics:
 *   blue   = CPU scalar / framework
 *   amber  = CPU SIMD (AVX2/FMA, NEON)  
 *   green  = GPU (PTX kernels via purego CUDA)
 *   red    = GPU (NV direct ioctl, zero deps)
 *   purple = both CPU + GPU (DevBuf dispatch)
 *
 * Style: taoofmac.com portfolio dark/light responsive
 *
 * @module render-test-matrix
 * @kind entrypoint
 */

import { writeFileSync } from "fs";
import { resolve } from "path";

// --- Data model ---
type Status = "pass" | "warn" | "skip" | "na";
interface Row {
  name: string;
  cpu: Status; gpu: Status; nv: Status;
  target: "cpu" | "simd" | "gpu" | "ioctl" | "both"; // colour hint
  tests: string; perf: string; note: string;
}
interface Section { title: string; rows: Row[]; }

const sections: Section[] = [
  {
    title: "Core Framework (tensor/) — CPU scalar",
    rows: [
      { name: "Tensor DAG + Realize",       cpu: "pass", gpu: "na",   nv: "na", target: "cpu",  tests: "48", perf: "—",              note: "Lazy eval" },
      { name: "Elementwise Fusion",          cpu: "pass", gpu: "na",   nv: "na", target: "cpu",  tests: "✓",  perf: "2× chained ops", note: "Fuse engine" },
      { name: "Pattern Rewrite (16 rules)",  cpu: "pass", gpu: "na",   nv: "na", target: "cpu",  tests: "✓",  perf: "const fold",     note: "tinygrad-style" },
      { name: "NumPy Reference Tests",       cpu: "pass", gpu: "na",   nv: "na", target: "cpu",  tests: "20", perf: "&lt;1e-5 diff",     note: "Ground truth" },
    ]
  },
  {
    title: "Safetensors (safetensors/) — CPU",
    rows: [
      { name: "F16/BF16/F32 + Sharded",     cpu: "pass", gpu: "na",   nv: "na", target: "cpu",  tests: "3",  perf: "HuggingFace",    note: "Up to 15GB" },
      { name: "GPTQ INT4 Dequantization",    cpu: "pass", gpu: "na",   nv: "na", target: "simd", tests: "✓",  perf: "∥ 30s→14s",      note: "AutoRound" },
    ]
  },
  {
    title: "Model Inference (model/) — CPU SIMD + GPU",
    rows: [
      { name: "BERT Encoder (GTE-small)",    cpu: "pass", gpu: "na",   nv: "na", target: "simd", tests: "6",  perf: "10.8ms, 0 alloc", note: "AVX2 SGEMM" },
      { name: "LLaMA (SmolLM2-135M)",        cpu: "pass", gpu: "warn", nv: "na", target: "both", tests: "✓",  perf: "CPU 35 GPU 50ms", note: "h=576" },
      { name: "LLaMA (Qwen2.5-7B INT4)",     cpu: "pass", gpu: "pass", nv: "na", target: "both", tests: "✓",  perf: "CPU 1005 GPU 518ms", note: "1.9× speedup" },
      { name: "BPE Tokenizer",               cpu: "pass", gpu: "na",   nv: "na", target: "cpu",  tests: "✓",  perf: "GPT-2 + Qwen",   note: "Str + array" },
    ]
  },
  {
    title: "GPU Compute (gpu/) — 10 PTX kernels + purego CUDA",
    rows: [
      { name: "SGEMM (16×16 tiled PTX)",     cpu: "pass", gpu: "pass", nv: "na", target: "gpu",  tests: "1",  perf: "348 GFLOPS",     note: "Shared mem" },
      { name: "INT4 Fused Dequant+GEMV",      cpu: "pass", gpu: "pass", nv: "na", target: "gpu",  tests: "✓",  perf: "197µs 3584²",    note: "8× unroll" },
      { name: "vec_add / vec_mul / vec_scale", cpu: "pass", gpu: "pass", nv: "na", target: "both", tests: "4",  perf: "Thresh ≥2048",   note: "Auto fallback" },
      { name: "vec_silu (SiLU activation)",    cpu: "pass", gpu: "pass", nv: "na", target: "both", tests: "✓",  perf: "exp2 approx",    note: "x·σ(x)" },
      { name: "rms_norm (shared mem reduce)",  cpu: "pass", gpu: "pass", nv: "na", target: "both", tests: "✓",  perf: "256-thread",     note: "rsqrt approx" },
      { name: "RoPE (cos/sin approx PTX)",    cpu: "pass", gpu: "pass", nv: "na", target: "gpu",  tests: "✓",  perf: "cos.approx",     note: "In-place rotate" },
      { name: "GQA Attention (per-head PTX)",  cpu: "pass", gpu: "pass", nv: "na", target: "gpu",  tests: "✓",  perf: "softmax+V weight", note: "Shared scores[]" },
      { name: "DevBuf (CPU↔GPU dispatch)",     cpu: "pass", gpu: "pass", nv: "na", target: "both", tests: "4",  perf: "Lazy, MarkDirty", note: "tinygrad" },
    ]
  },
  {
    title: "NV Direct ioctl (gpu/) — zero dependencies",
    rows: [
      { name: "Device discovery + UUID",       cpu: "na",   gpu: "na", nv: "pass", target: "ioctl", tests: "1", perf: "RTX 3060",        note: "Raw syscall" },
      { name: "GPU caps (84 SMs, sm_86)",      cpu: "na",   gpu: "na", nv: "pass", target: "ioctl", tests: "1", perf: "7×6×2 SM",        note: "RM control" },
      { name: "Channel group + Ctx share",     cpu: "na",   gpu: "na", nv: "pass", target: "ioctl", tests: "1", perf: "KEPLER+FERMI",    note: "Compute path" },
      { name: "Host memory registration",      cpu: "na",   gpu: "na", nv: "pass", target: "ioctl", tests: "1", perf: "NVOS02 + mmap",   note: "CPU r/w" },
      { name: "VA space + UVM",                cpu: "na",   gpu: "na", nv: "warn", target: "ioctl", tests: "—", perf: "FERMI_VASPACE",   note: "No page fault" },
    ]
  },
];

// --- Palette: target → row background tint ---
const ROW_TINT_LIGHT: Record<string, string> = {
  cpu:   "#f8fafc", // neutral
  simd:  "#fffbeb", // amber tint
  gpu:   "#f0fdf4", // green tint
  ioctl: "#fef2f2", // red tint
  both:  "#f5f3ff", // purple tint
};
const ROW_TINT_DARK: Record<string, string> = {
  cpu:   "#0f1218",
  simd:  "#1a1808",
  gpu:   "#0a1a0e",
  ioctl: "#1a0c0c",
  both:  "#140e1e",
};

// --- Layout ---
const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

const W = 920, ROW_H = 22, SEC_H = 20, HDR_H = 26, PAD = 8;
const COL = { name: 24, cpu: 230, gpu: 310, nv: 440, tests: 555, perf: 610, note: 800 };

let totalH = 72;
for (const s of sections) totalH += SEC_H + 4 + s.rows.length * ROW_H + 4;
totalH += 40;

// --- Build SVG ---
let svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${totalH}" font-family="system-ui,-apple-system,sans-serif">\n`;

svg += `  <style>
    @media (prefers-color-scheme: dark) {
      .bg { fill: #0f1218; }
      .hdr-bg { fill: #1e293b; }
      .sec-bg { fill: #1a1e28; stroke: #2a3040; stroke-width: 0.5; }
      .hdr-text { fill: #e2e8f0; font-size: 11px; font-weight: 600; }
      .cell { fill: #cbd5e1; font-size: 10.5px; }
      .note { fill: #64748b; font-size: 9.5px; }
      .perf { fill: #86efac; font-size: 10px; font-family: "SF Mono", Menlo, monospace; }
      .title { fill: #f1f5f9; font-size: 18px; font-weight: 700; }
      .subtitle { fill: #64748b; font-size: 11px; }
      .sec-title { fill: #94a3b8; font-size: 11.5px; font-weight: 600; }
      .pass { fill: #22c55e; } .warn { fill: #f59e0b; }
      .skip { fill: #475569; } .na { fill: #334155; }
      .footer-bg { fill: #1e293b; }
      .footer-text { fill: #94a3b8; font-size: 11px; font-weight: 600; }
      .row-cpu   { fill: ${ROW_TINT_DARK.cpu}; }
      .row-simd  { fill: ${ROW_TINT_DARK.simd}; }
      .row-gpu   { fill: ${ROW_TINT_DARK.gpu}; }
      .row-ioctl { fill: ${ROW_TINT_DARK.ioctl}; }
      .row-both  { fill: ${ROW_TINT_DARK.both}; }
      .legend-cpu  { fill: #334155; stroke: #475569; }
      .legend-simd { fill: #422006; stroke: #a06020; }
      .legend-gpu  { fill: #052e16; stroke: #16a34a; }
      .legend-ioctl { fill: #2a0a0a; stroke: #a03030; }
      .legend-both { fill: #1e0a3a; stroke: #7c3aed; }
    }
    @media (prefers-color-scheme: light) {
      .bg { fill: #f8fafc; }
      .hdr-bg { fill: #334155; }
      .sec-bg { fill: #e2e8f0; stroke: #cbd5e1; stroke-width: 0.5; }
      .hdr-text { fill: #ffffff; font-size: 11px; font-weight: 600; }
      .cell { fill: #1e293b; font-size: 10.5px; }
      .note { fill: #64748b; font-size: 9.5px; }
      .perf { fill: #15803d; font-size: 10px; font-family: "SF Mono", Menlo, monospace; }
      .title { fill: #0f172a; font-size: 18px; font-weight: 700; }
      .subtitle { fill: #64748b; font-size: 11px; }
      .sec-title { fill: #475569; font-size: 11.5px; font-weight: 600; }
      .pass { fill: #22c55e; } .warn { fill: #f59e0b; }
      .skip { fill: #94a3b8; } .na { fill: #cbd5e1; }
      .footer-bg { fill: #1e293b; }
      .footer-text { fill: #e2e8f0; font-size: 11px; font-weight: 600; }
      .row-cpu   { fill: ${ROW_TINT_LIGHT.cpu}; }
      .row-simd  { fill: ${ROW_TINT_LIGHT.simd}; }
      .row-gpu   { fill: ${ROW_TINT_LIGHT.gpu}; }
      .row-ioctl { fill: ${ROW_TINT_LIGHT.ioctl}; }
      .row-both  { fill: ${ROW_TINT_LIGHT.both}; }
      .legend-cpu  { fill: #e2e8f0; stroke: #94a3b8; }
      .legend-simd { fill: #fef3c7; stroke: #d97706; }
      .legend-gpu  { fill: #dcfce7; stroke: #16a34a; }
      .legend-ioctl { fill: #fee2e2; stroke: #dc2626; }
      .legend-both { fill: #ede9fe; stroke: #7c3aed; }
    }
    text { font-family: -apple-system, "Segoe UI", Helvetica, sans-serif; }
  </style>\n`;

// Background
svg += `  <rect width="${W}" height="${totalH}" rx="10" class="bg"/>\n`;

// Title
svg += `  <text x="24" y="28" class="title">go-tinygrad Test Matrix</text>\n`;
svg += `  <text x="24" y="44" class="subtitle">Pure Go + PTX · RTX 3060 12GB · 7 packages · 67 tests · 10 kernels</text>\n`;

// Legend — execution target colours
const legendItems = [
  ["legend-cpu",   "CPU scalar"],
  ["legend-simd",  "CPU SIMD (AVX2/NEON)"],
  ["legend-gpu",   "GPU PTX (purego)"],
  ["legend-ioctl", "GPU NV ioctl"],
  ["legend-both",  "CPU + GPU (DevBuf)"],
];
let lx = 460;
for (const [cls, lbl] of legendItems) {
  svg += `  <rect x="${lx}" y="20" width="10" height="10" rx="2" class="${cls}" stroke-width="1.5"/><text x="${lx+14}" y="29" class="note">${lbl}</text>\n`;
  lx += 12 + lbl.length * 5.5 + 8;
}

// Status legend
svg += `  <g transform="translate(460,38)">`;
for (const [i, [cls, lbl]] of ([ ["pass","Pass"], ["warn","Partial"], ["na","N/A"] ] as const).entries()) {
  svg += `<circle cx="${i*65}" cy="0" r="4" class="${cls}"/><text x="${i*65+8}" y="4" class="note">${lbl}</text>`;
}
svg += `</g>\n`;

// Column headers
let y = 56;
svg += `  <rect x="0" y="${y}" width="${W}" height="${HDR_H}" rx="0" class="hdr-bg"/>\n`;
svg += `  <text x="${COL.name}" y="${y+17}" class="hdr-text">Component</text>`;
svg += `<text x="${COL.cpu}" y="${y+17}" class="hdr-text">CPU</text>`;
svg += `<text x="${COL.gpu}" y="${y+17}" class="hdr-text">GPU (purego)</text>`;
svg += `<text x="${COL.nv}" y="${y+17}" class="hdr-text">GPU (NV ioctl)</text>`;
svg += `<text x="${COL.tests}" y="${y+17}" class="hdr-text">Tests</text>`;
svg += `<text x="${COL.perf}" y="${y+17}" class="hdr-text">Performance</text>`;
svg += `<text x="${COL.note}" y="${y+17}" class="hdr-text">Notes</text>\n`;
y += HDR_H;

// Sections + rows
for (const sec of sections) {
  y += 4;
  svg += `  <rect x="${PAD}" y="${y}" width="${W - PAD*2}" height="${SEC_H}" rx="3" class="sec-bg"/>\n`;
  svg += `  <text x="16" y="${y + 14}" class="sec-title">${esc(sec.title)}</text>\n`;
  y += SEC_H;

  for (const row of sec.rows) {
    // Row background tint by target
    svg += `  <rect x="${PAD}" y="${y}" width="${W - PAD*2}" height="${ROW_H}" rx="2" class="row-${row.target}"/>\n`;
    svg += `  <text x="${COL.name}" y="${y+15}" class="cell">${esc(row.name)}</text>`;
    svg += `<circle cx="${COL.cpu + 16}" cy="${y+11}" r="5" class="${row.cpu}"/>`;
    svg += `<circle cx="${COL.gpu + 40}" cy="${y+11}" r="5" class="${row.gpu}"/>`;
    svg += `<circle cx="${COL.nv + 40}" cy="${y+11}" r="5" class="${row.nv}"/>`;
    svg += `<text x="${COL.tests}" y="${y+15}" class="perf">${esc(row.tests)}</text>`;
    svg += `<text x="${COL.perf}" y="${y+15}" class="perf">${esc(row.perf)}</text>`;
    svg += `<text x="${COL.note}" y="${y+15}" class="note">${esc(row.note)}</text>\n`;
    y += ROW_H;
  }
}

// Footer
y += 8;
svg += `  <rect x="${PAD}" y="${y}" width="${W - PAD*2}" height="28" rx="6" class="footer-bg"/>\n`;
svg += `  <text x="24" y="${y+18}" class="footer-text">67 tests · 7.2K lines Go · 10 PTX kernels · tensor 72% · model 61% · gpu 23% · 7B GPU 518ms (1.9×) · 135M CPU 36ms (28 tok/s)</text>\n`;

svg += `</svg>\n`;

// --- Output ---
const outPath = process.argv.includes("--output")
  ? process.argv[process.argv.indexOf("--output") + 1]
  : resolve(__dirname, "../docs/test-matrix.svg");

writeFileSync(outPath, svg);
console.log(`Wrote ${outPath} (${svg.length} bytes)`);
