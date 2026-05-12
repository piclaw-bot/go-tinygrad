# SIMD folder reorg notes

Phase 6.6 tracks the cleanup of `backends/simd` after the package move from the old root `simd` package.

## Current layout

`backends/simd` is intentionally still a single Go package. Go build constraints already split most implementation files by CPU family:

- shared facade and scalar fallback:
  - `simd.go`
  - `vec.go`
  - `bf16.go`
  - `sgemm.go`
  - `sgemm_blocked.go`
  - `gebp.go`
  - `pack.go`
  - `capabilities.go`
- amd64-specific implementation:
  - `*_amd64.go`
  - `*_amd64.s`
  - `*_amd64_dispatch.go`
- arm64-specific implementation:
  - `*_arm64.go`
  - `*_arm64.s`
  - `*_arm64_dispatch.go`
- portable fallback implementation:
  - `*_other.go`

## Constraint

A literal folder split (`backends/simd/amd64`, `backends/simd/arm64`, etc.) would create separate Go packages. That is possible, but it is not a purely mechanical move: the public `backends/simd` package would need to become a facade over internal CPU-family packages, and some unexported helper/assembly entrypoints would need explicit bridge APIs.

Because Phase 6.5/6.6 is still guarding against accidental semantic churn, the safe migration path is:

1. Keep `backends/simd` as the public package boundary.
2. Split oversized mixed-concern files inside that package first.
3. Move architecture-specific implementations behind narrow internal packages only after wrapper APIs are explicit and tests are stable.

## Target shape

A future package split should look like this:

```text
backends/simd/
  facade.go              # public Sdot/Saxpy/Vec*/RMSNorm/etc.
  scalar/                # pure-Go fallback helpers, no CPU feature dependency
  amd64/                 # AVX2/FMA assembly and wrappers
  arm64/                 # NEON assembly and wrappers
  internal/cpufeatures/  # runtime detection glue if needed
```

The public import path should remain `github.com/rcarmo/go-pherence/backends/simd`.

## Safe next mechanical steps

Before introducing subpackages:

- keep scalar fallback functions small and explicitly named (dot/SAXPY fallbacks now live in `scalar.go`);
- ensure each public wrapper has scalar fallback coverage and precise scalar math for norm-sensitive paths;
- keep architecture-specific build tags on all assembly-backed entrypoints;
- avoid moving unexported assembly symbols across package boundaries until the public facade wrappers are complete;
- keep shared guard helpers such as `checkedMulInt` in the facade package until subpackage bridge APIs are designed;
- run at minimum:

```sh
go test ./backends/simd -count=1
go test ./tensor ./backends/simd ./runtime/... ./loader/... -count=1
go test ./... -run '^$'
go vet ./...
git diff --check
```
