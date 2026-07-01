# Changelog

All notable changes to Tensotron are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0-alpha] - 2026-07-01

### Added
- First public release, published to NuGet as `Tensotron`.
- PyTorch-faithful, float32 tensor + autograd engine for .NET: define-by-run autograd, a broad
  torch-parity-tested op surface, `Module`/`Sequential`/`Linear`, SGD/Adam/AdamW/RMSProp, LR
  schedulers, `DataLoader`, and full-checkpoint save/load.
- CUDA backend via ILGPU + cuBLAS SGEMM; a hand-written managed/SIMD CPU backend
  (`TENSOTRON_BACKEND=simd`) as the GPU-less fallback and the fast small-net inference path.

[Unreleased]: https://github.com/mfagerlund/Tensotron/compare/v0.1.0-alpha...HEAD
[0.1.0-alpha]: https://github.com/mfagerlund/Tensotron/releases/tag/v0.1.0-alpha
