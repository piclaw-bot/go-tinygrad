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
- keep shared guard helpers such as `checkedMulInt`/`checkedAddInt` in the facade package until subpackage bridge APIs are designed;
- keep SGEMM/GEBP/gather capability checks (`HasSgemmAsm`) at the public facade boundary so unsupported architectures no-op safely instead of reaching fallback panic stubs;
- keep blocked/GEBP/gather SGEMM pointer arithmetic behind checked shape products and checked float32 byte offsets before `unsafe.Add`;
- keep GEBP packing scratch per-call rather than package-global so concurrent facade calls cannot alias packed B tiles;
- route zero-length vector/BF16 entrypoints through scalar fallbacks instead of assembly stubs;
- run at minimum:

```sh
go test ./backends/simd -count=1
go test ./tensor ./backends/simd ./runtime/... ./loader/... -count=1
go test ./... -run '^$'
go vet ./...
git diff --check
```

## Bridge API design before subpackages

The literal `backends/simd/{amd64,arm64,scalar}` split should not start until the facade has explicit bridge contracts for each implementation family. The bridge should keep the public import path stable and make architecture packages dumb providers of kernels, not owners of shape validation or fallback policy.

### Facade responsibilities that must stay in `backends/simd`

- Export the public API (`Sdot`, `Saxpy`, `Vec*`, `RMSNorm*`, `BF16*`, `Sgemm*`, `Pack*`, GEBP/gather helpers).
- Own all public input validation, nil/length/stride checks, overflow checks, and scalar fallback selection.
- Own runtime feature detection and the public `RuntimeCapabilities`, `HasDotAsm`, `HasVecAsm`, and `HasSgemmAsm` compatibility variables.
- Own scalar fallback behavior for malformed, zero-length, or partially sized inputs so architecture providers can assume prevalidated exact-shape calls.
- Keep `checkedMulInt`/`checkedAddInt` and SGEMM/GEBP/gather preflight helpers at the facade boundary until every provider API has equivalent tests.

### Provider shape

Future architecture packages should expose a narrow provider value rather than public free functions used directly by model code:

```go
type Kernels struct {
    Caps Capabilities

    Dot   DotKernels
    Vec   VecKernels
    BF16  BF16Kernels
    SGEMM SGEMMKernels
    Pack  PackKernels
}

type DotKernels struct {
    Sdot  func(x, y []float32) float32
    Saxpy func(a float32, x, y []float32)
}

type VecKernels struct {
    Snrm2          func(x []float32) float32
    Add            func(dst, a, b []float32)
    Mul            func(dst, a, b []float32)
    Scale          func(dst, a []float32, scale float32)
    ScaleAdd       func(dst, a, b []float32, scale float32)
    SiLUMul        func(dst, a, b []float32)
    GELUTanhMul    func(dst, a, b []float32)
    RMSNorm        func(x, w []float32, eps float32)
    RMSNormBF16    func(x, w []float32, eps float32)
    RMSNormNoScale func(x []float32, eps float32)
    ToBF16         func(x []float32)
}

type BF16Kernels struct {
    Dot       func(x, y []uint16) float32
    Add       func(dst, a, b []uint16)
    RMSNorm   func(x, w []uint16, eps float32)
    WidenF32  func(dst []float32, src []uint16)
    NarrowF32 func(dst []uint16, src []float32)
}

type SGEMMKernels struct {
    NN func(m, n, k int, alpha float32, a, b, c unsafe.Pointer, lda, ldb, ldc int)
    NT func(m, n, k int, alpha float32, a, b, c unsafe.Pointer, lda, ldb, ldc int)
}
```

The facade should select one provider at init/runtime capability time:

- `scalar.Provider()` is always available and contains only pure-Go fallbacks.
- `amd64.Provider()` is selected only when `runtime.GOARCH == "amd64"` and AVX2/FMA capability gates pass.
- `arm64.Provider()` is selected only when `runtime.GOARCH == "arm64"` and ASIMD/NEON capability gates pass.

### Assembly bridge rules

- Assembly symbols stay package-local to their provider package; the facade must not call `snrm2Asm`/`vecAddAsm`-style names directly after the split.
- Providers may assume exact lengths and valid pointers only when the facade calls an `Unsafe`/prevalidated kernel. Public wrappers keep defensive partial-length/no-op behavior.
- SGEMM/GEBP/gather providers must not panic on unsupported architectures through public entrypoints. The facade continues to gate via capabilities and should route unsupported calls to safe no-op/error-free fallback behavior used today.
- Norm-sensitive scalar paths must keep precise `math.Sqrt` behavior; SIMD approximations require tests that bound drift against the scalar reference.

### Migration order

1. Introduce internal provider structs inside the current single package and route public wrappers through them without changing exported APIs.
2. Add tests that compare the selected provider against scalar fallbacks for vectors, BF16, SGEMM/GEBP/gather bounds, and malformed input behavior.
3. Move scalar provider files into `backends/simd/scalar` and keep facade wrappers/tests unchanged.
4. Move amd64 and arm64 providers one family at a time, preserving build tags and assembly symbol visibility.
5. Only after all providers are split, remove transitional package-level assembly declarations from the facade.

This design keeps `github.com/rcarmo/go-pherence/backends/simd` as the only public import path while giving future subpackages a clear implementation-only contract.


## Recent facade guard baseline

Recent Phase 6.6 audit fixes now also require provider APIs to preserve these facade behaviors:

- Public vector/BF16 entrypoints route zero-length slices through scalar fallbacks rather than assembly stubs.
- GEBP packing scratch is per-call, not package-global, so concurrent calls cannot alias packed B tiles.
- Unsupported-architecture public SGEMM entrypoints are safe no-ops behind `HasSgemmAsm`; private provider stubs may still panic if reached incorrectly.
- Blocked/GEBP/gather SGEMM pointer offsets use checked products and checked float32 byte offsets before `unsafe.Add`.
