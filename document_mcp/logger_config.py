import logging
from pathlib import Path
import functools # Added import

# --- Logging Setup ---
mcp_call_logger = logging.getLogger('mcp_call_logger')
mcp_call_logger.setLevel(logging.INFO)

# Use Path(__file__).resolve().parent to ensure the path is correct even if the script is called from elsewhere.
log_file_path = Path(__file__).resolve().parent / 'mcp_calls.log'
file_handler = logging.FileHandler(log_file_path)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
mcp_call_logger.addHandler(file_handler)
# Prevent logs from propagating to the root logger if not desired
mcp_call_logger.propagate = False

# --- Decorator for Logging MCP Calls ---
def log_mcp_call(func):
    @functools.wraps(func) # Use functools.wraps to preserve function metadata
    def wrapper(*args, **kwargs):
        # Try to get the actual function name if it's a bound method or has other wrappers
        func_name = getattr(func, '__name__', 'unknown_function')

        try:
            logged_args = []
            for arg in args:
                if hasattr(arg, 'model_dump_json'): # Check for Pydantic v2 model
                    logged_args.append(arg.model_dump_json(indent=None, exclude_none=True)) # Compact JSON
                elif hasattr(arg, 'json'): # Check for Pydantic v1 model
                    logged_args.append(arg.json(indent=None, exclude_none=True)) # Compact JSON
                else:
                    logged_args.append(repr(arg))

            logged_kwargs = {}
            for k, v in kwargs.items():
                if hasattr(v, 'model_dump_json'):
                    logged_kwargs[k] = v.model_dump_json(indent=None, exclude_none=True)
                elif hasattr(v, 'json'):
                    logged_kwargs[k] = v.json(indent=None, exclude_none=True)
                else:
                    logged_kwargs[k] = repr(v)

            arg_str = f"args={logged_args}, kwargs={logged_kwargs}"
        except Exception as e:
            arg_str = f"args/kwargs logging error: {e}"

        mcp_call_logger.info(f"Calling tool: {func_name} with {arg_str}")
        try:
            result = func(*args, **kwargs)

            try:
                if isinstance(result, list):
                    # Check if it's a list of Pydantic-like objects
                    if result and (hasattr(result[0], 'model_dump_json') or hasattr(result[0], 'json')):
                        logged_list = []
                        for item in result:
                            if hasattr(item, 'model_dump_json'):
                                logged_list.append(item.model_dump_json(indent=None, exclude_none=True))
                            elif hasattr(item, 'json'): # Pydantic v1 fallback
                                logged_list.append(item.json(indent=None, exclude_none=True))
                            else:
                                logged_list.append(repr(item))
                        result_str = "[" + ", ".join(logged_list) + "]"
                    else:
                        # It's a list, but not of Pydantic models (or empty)
                        result_str = repr(result)
                elif hasattr(result, 'model_dump_json'): # Single Pydantic v2 model
                    result_str = result.model_dump_json(indent=None, exclude_none=True)
                elif hasattr(result, 'json'): # Single Pydantic v1 model
                    result_str = result.json(indent=None, exclude_none=True)
                else: # Other types
                    result_str = repr(result)
                # Removed the 500-character truncation
            except Exception as e:
                result_str = f"Result logging error: {e}"

            mcp_call_logger.info(f"Tool {func_name} returned: {result_str}")
            return result
        except Exception as e:
            mcp_call_logger.error(f"Tool {func_name} raised exception: {e}", exc_info=True)
            raise
    return wrapper
