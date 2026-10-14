"""Compatibility import for the standalone E2B relay."""

from reef.sandbox.e2b_relay import Relay, main

__all__ = ["Relay", "main"]

if __name__ == "__main__":
    main()
