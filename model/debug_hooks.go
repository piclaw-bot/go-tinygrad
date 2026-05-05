package model

// Optional internal trace hooks for diagnostics/tests.
// Nil in normal execution.
var debugLayerHook func(backend string, step, layer int, hidden []float32)
var debugLogitsHook func(backend string, step int, hidden, logits []float32)
var debugOpHook func(backend string, step, layer int, op string, vec []float32)
