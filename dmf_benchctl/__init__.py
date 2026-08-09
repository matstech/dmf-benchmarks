"""Host-side orchestration control plane for DMF Benchmarks."""

from importlib import metadata


try:
    __version__ = metadata.version("dmf-benchmarks")
except metadata.PackageNotFoundError:
    __version__ = "0.2.0"
