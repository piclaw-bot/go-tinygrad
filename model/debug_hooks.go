package model

// Optional internal trace hooks for diagnostics/tests.
// Nil in normal execution.
var debugLayerHook func(backend string, step, layer int, hidden []float32)
var debugLogitsHook func(backend string, step int, hidden, logits []float32)
var debugOpHook func(backend string, step, layer int, op string, vec []float32)

// Optional CPU-only hidden-state mutator used by diagnostics to replay later layers
// from a substituted hidden state without duplicating the whole forward path.
var debugCPUHiddenInOverrideHook func(step, layer int, hidden []float32)

// Optional CPU-only per-layer-input mutator used by diagnostics to replay PLI
// inputs from substituted values without duplicating the whole front-end path.
var debugCPUPerLayerInputsOverrideHook func(step int, perLayerInputs [][]float32)

// Optional CPU-only MLP-input mutator used by diagnostics to replay from a
// substituted post-attention mlp_input without duplicating the full layer path.
var debugCPUMLPInputOverrideHook func(step, layer int, mlpInput []float32)
