"""Performative Test-Time Adaptation (PerfTTA) reference implementation.

Core idea: in online TTA the adapting model INDUCES the distribution it next
adapts to. This package implements the performative loop, a controllable
feedback (distribution) map, baseline adapters (Source/TENT/EATA-lite), the
performativity-aware adapter (PATTA), and collapse/stability metrics.
"""
__version__ = "0.1.0"
