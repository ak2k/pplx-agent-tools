"""Verb layer: per-verb request shape + response parsing → typed Result.

Each verb owns the knowledge of which Perplexity endpoint to call, what to
send, and how to parse the response into a typed Result. The transport seam
(`wire.Client`) is passed in so tests can swap a fake.
"""
