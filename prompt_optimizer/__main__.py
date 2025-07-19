"""Main entry point for the prompt optimizer package.
"""

import asyncio

from .cli import main

if __name__ == "__main__":
    asyncio.run(main())
