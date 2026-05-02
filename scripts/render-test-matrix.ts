#!/usr/bin/env bun
/**
 * render-test-matrix.ts — Generate the go-tinygrad test matrix SVG.
 *
 * Usage: bun run scripts/render-test-matrix.ts [--output docs/test-matrix.svg]
 *
 * @module render-test-matrix
 * @kind entrypoint
 */

import { writeFileSync } from "fs";
import { resolve } from "path";

// Status symbols
const PASS = `<circle cx="0" cy="0" r="6" class="pass"/>`;
const WARN = `<circle cx="0" cy="0" r="6" class="warn"/>`;
const SKIP = `<circle cx="0" cy="0" r="6" class="skip"/>`;
const NA   = `<circle cx="0" cy="0" r="6" class="na"/>`;

interface Row {
  name: string;
  cpu: string; gpu: string; nv: string;
  tests: string; perf: string; note: string;
}

interface Section {
  title: string;
  rows: Row[];
}

const sections: Section[] = [
  {
    title: "Core Framework (tensor/)",
    rows: [
      { name: "Tensor DAG + Realize",       cpu: "pass", gpu: "na", nv: "na", tests: "48", perf: "—",              note: "Lazy eval" },
      { name: "Elementwise Fusion",          cpu: "pass", gpu: "na", nv: "na", tests: "✓",  perf: "2× chained ops", note: "Fuse engine" },
      { name: "Pattern Rewrite (16 rules)",  cpu: "pass", gpu: "na", nv: "na", tests: "✓",  perf: "const fold",     note: "tinygrad-style" },
      { name: "NumPy Reference Tests",       cpu: "pass", gpu: "na", nv: "na", tests: "20", perf: "<1e-5 diff",     note: "Ground truth" },
    ]
  },
  {
    title: "Safetensors (safetensors/)",
    rows: [
      { name: "F16/BF16/F32 + Sharded",     cpu: "pass", gpu: "na", nv: "na", tests: "3",  perf: "HuggingFace",    note: "Up to 15GB" },
      { name: "GPTQ INT4 Dequantization",    cpu: "pass", gpu: "na", nv: "na", tests: "✓",  perf: "∥ 30s→14s",      note: "AutoRound" },
    ]
  },
  {
    title: "Model Inference (model/)",
    rows: [
      { name: "BERT Encoder (GTE-small)",    cpu: "pass", gpu: "na", nv: "na", tests: "6",  perf: "10.8ms, 0 alloc", note: "gte-go parity" },
      { name: "LLaMA (SmolLM2-135M)",        cpu: "pass", gpu: "warn", nv: "na", tests: "✓", perf: "CPU 35 GPU 50ms", note: "h=576" },
      { name: "LLaMA (Qwen2.5-7B INT4)",     cpu: "pass", gpu: "warn", nv: "na", tests: "✓", perf: "CPU 1009ms/tok",  note: "GPTQ 5GB" },
      { name: "BPE Tokenizer",               cpu: "pass", gpu: "na", nv: "na", tests: "✓",  perf: "GPT-2 + Qwen",   note: "Str + array" },
    ]
  },
  {
    title: "GPU Compute (gpu/)",
    rows: [
      { name: "SGEMM (16×16 tiled PTX)",     cpu: "pass", gpu: "pass", nv: "na", tests: "4", perf: "348 GFLOPS",     note: "Shared mem" },
      { name: "INT4 Fused Dequant+GEMV",      cpu: "pass", gpu: "pass", nv: "na", tests: "✓", perf: "197µs 3584²",    note: "8× unroll" },
      { name: "vec_add / vec_mul / vec_scale", cpu: "pass", gpu: "pass", nv: "na", tests: "4", perf: "Thresh ≥2048",   note: "Auto fallback" },
      { name: "vec_silu (SiLU activation)",    cpu: "pass", gpu: "pass", nv: "na", tests: "✓", perf: "exp2 approx",    note: "x·σ(x)" },
      { name: "rms_norm (shared mem reduce)",  cpu: "pass", gpu: "pass", nv: "na", tests: "✓", perf: "256-thread",     note: "rsqrt approx" },
      { name: "DevBuf (CPU↔GPU)",              cpu: "pass", gpu: "pass", nv: "na", tests: "4", perf: "Lazy, MarkDirty", note: "tinygrad" },
    ]
  },
  {
    title: "NV Direct ioctl (gpu/ — zero deps)",
    rows: [
      { name: "Device discovery + UUID",       cpu: "na", gpu: "na", nv: "pass", tests: "1", perf: "RTX 3060",        note: "Raw syscall" },
      { name: "GPU caps (84 SMs, sm_86)",      cpu: "na", gpu: "na", nv: "pass", tests: "1", perf: "7×6×2 SM",        note: "RM control" },
      { name: "Channel group + Ctx share",     cpu: "na", gpu: "na", nv: "pass", tests: "1", perf: "KEPLER+FERMI",    note: "Compute path" },
      { name: "Host memory registration",      cpu: "na", gpu: "na", nv: "pass", tests: "1", perf: "NVOS02 + mmap",   note: "CPU r/w" },
      { name: "VA space + UVM",                cpu: "na", gpu: "na", nv: "warn", tests: "—", perf: "FERMI_VASPACE",   note: "No page fault" },
    ]
  },
];

// --- Render ---
const W = 920, ROW_H = 20, SEC_H = 18, HDR_H = 24;
const COL = { name: 24, cpu: 220, gpu: 300, nv: 420, tests: 555, perf: 610, note: 800 };

let totalH = 70; // title
for (const s of sections) totalH += SEC_H + 4 + s.rows.length * ROW_H + 4;
totalH += 36; // footer

const statusIcon = (s: string, cx: number, cy: number) =>
  `<circle cx="${cx}" cy="${cy}" r="6" class="${s}"/>`;

let svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${totalH}" font-family="system-ui,-apple-system,sans-serif">\n`;
svg += `  <style>
    @media (prefers-color-scheme: dark) {
      .bg { fill: #0f1218; } .hdr-bg { fill: #1e293b; } .sec-bg { fill: #1a1e28; }
      .hdr { fill: #e2e8f0; } .cell { fill: #cbd5e1; } .note { fill: #64748b; }
      .perf { fill: #86efac; } .title { fill: #f1f5f9; } .sub { fill: #64748b; }
      .pass { fill: #22c55e; } .fail { fill: #ef4444; } .warn { fill: #f59e0b; }
      .skip { fill: #475569; } .na { fill: #334155; }
      .footer { fill: #1e293b; } .footer-text { fill: #94a3b8; }
    }
    @media (prefers-color-scheme: light) {
      .bg { fill: #f8fafc; } .hdr-bg { fill: #334155; } .sec-bg { fill: #e2e8f0; }
      .hdr { fill: #ffffff; } .cell { fill: #1e293b; } .note { fill: #64748b; }
      .perf { fill: #15803d; } .title { fill: #0f172a; } .sub { fill: #64748b; }
      .pass { fill: #22c55e; } .fail { fill: #ef4444; } .warn { fill: #f59e0b; }
      .skip { fill: #94a3b8; } .na { fill: #cbd5e1; }
      .footer { fill: #1e293b; } .footer-text { fill: #e2e8f0; }
    }
    text { font-family: -apple-system, "Segoe UI", Helvetica, sans-serif; }
    .title { font-size: 18px; font-weight: 700; }
    .sub { font-size: 11px; }
    .hdr { font-size: 11px; font-weight: 600; }
    .cell { font-size: 10.5px; }
    .note { font-size: 9.5px; }
    .perf { font-size: 10px; font-family: "SF Mono", Menlo, monospace; }
    .sec { font-size: 12px; font-weight: 600; }
    .footer-text { font-size: 11px; font-weight: 600; }
  </style>\n`;

svg += `  <rect width="${W}" height="${totalH}" rx="10" class="bg"/>\n`;
svg += `  <text x="24" y="30" class="title">go-tinygrad Test Matrix</text>\n`;
svg += `  <text x="24" y="46" class="sub">Pure Go + PTX · RTX 3060 12GB · 7 packages · 66 tests · 25+ commits</text>\n`;

// Legend
svg += `  <g transform="translate(620,24)">`;
for (const [i, [cls, lbl]] of [["pass","Pass"],["warn","Partial"],["skip","Skip"],["na","N/A"]].entries()) {
  svg += `<circle cx="${i*70}" cy="0" r="5" class="${cls}"/><text x="${i*70+10}" y="4" class="cell">${lbl}</text>`;
}
svg += `</g>\n`;

// Header row
let y = 58;
svg += `  <rect x="0" y="${y}" width="${W}" height="${HDR_H}" class="hdr-bg"/>\n`;
svg += `  <text x="${COL.name}" y="${y+16}" class="hdr">Component</text>`;
svg += `<text x="${COL.cpu}" y="${y+16}" class="hdr">CPU</text>`;
svg += `<text x="${COL.gpu}" y="${y+16}" class="hdr">GPU (purego)</text>`;
svg += `<text x="${COL.nv}" y="${y+16}" class="hdr">GPU (NV ioctl)</text>`;
svg += `<text x="${COL.tests}" y="${y+16}" class="hdr">Tests</text>`;
svg += `<text x="${COL.perf}" y="${y+16}" class="hdr">Performance</text>`;
svg += `<text x="${COL.note}" y="${y+16}" class="hdr">Notes</text>\n`;
y += HDR_H;

// Sections
for (const sec of sections) {
  y += 4;
  svg += `  <rect x="8" y="${y}" width="${W-16}" height="${SEC_H}" rx="3" class="sec-bg"/>\n`;
  svg += `  <text x="16" y="${y+13}" class="sec cell">${sec.title}</text>\n`;
  y += SEC_H;

  for (const row of sec.rows) {
    svg += `  <text x="${COL.name}" y="${y+14}" class="cell">${row.name}</text>`;
    svg += statusIcon(row.cpu, COL.cpu + 20, y + 10);
    svg += statusIcon(row.gpu, COL.gpu + 40, y + 10);
    svg += statusIcon(row.nv, COL.nv + 40, y + 10);
    svg += `<text x="${COL.tests}" y="${y+14}" class="perf">${row.tests}</text>`;
    svg += `<text x="${COL.perf}" y="${y+14}" class="perf">${row.perf}</text>`;
    svg += `<text x="${COL.note}" y="${y+14}" class="note">${row.note}</text>\n`;
    y += ROW_H;
  }
}

// Footer
y += 8;
svg += `  <rect x="8" y="${y}" width="${W-16}" height="28" rx="6" class="footer"/>\n`;
svg += `  <text x="24" y="${y+18}" class="footer-text">66 tests · 7.0K lines Go · 8 PTX kernels · 2 GPU paths · RTX 3060 @ 348 GFLOPS · Qwen2.5-7B @ 1 tok/s · SmolLM2-135M @ 28 tok/s</text>\n`;

svg += `</svg>\n`;

const outPath = process.argv.includes("--output")
  ? process.argv[process.argv.indexOf("--output") + 1]
  : resolve(__dirname, "../docs/test-matrix.svg");

writeFileSync(outPath, svg);
console.log(`Wrote ${outPath} (${svg.length} bytes)`);
