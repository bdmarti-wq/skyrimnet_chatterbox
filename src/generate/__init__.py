"""
Package initialization for the audio generation subsystem.

This package contains the refactored generation pipeline that:
- Organizes generation into distinct phases
- Provides a unified cache management system
- Maintains compatibility with the 29-parameter UI interface

Key components:
- CacheManager: Centralized cache access
- GenerationPipeline: Orchestrates the audio generation process
- AudioGenerationContext: State object passed through pipeline
"""
# No direct imports here to prevent circular dependencies
# Clients should import specific modules they need

__all__ = []

