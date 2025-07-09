"""
Main entry point for the prompt optimizer package.
"""

from .cli import main
import asyncio

if __name__ == "__main__":
    asyncio.run(main())