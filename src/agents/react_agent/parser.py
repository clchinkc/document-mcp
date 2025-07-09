import json
from typing import Any, Dict, Tuple


class ActionParser:
    """
    Parses an action string in the format tool_name(arg1="value1", arg2="value2").
    """

    def parse(self, action_string: str) -> Tuple[str, Dict[str, Any]]:
        """
        Parses the action string and returns the tool name and arguments.

        Args:
            action_string: The action string to parse.

        Returns:
            A tuple containing the tool name and a dictionary of arguments.

        Raises:
            ValueError: If the action string is malformed.
        """
        action_string = action_string.strip()
        if not action_string.endswith(")"):
            raise ValueError("Action string must end with ')'")

        parts = action_string.split("(", 1)
        if len(parts) != 2:
            raise ValueError("Action string must contain '(' after tool name")

        tool_name = parts[0].strip()
        args_str = parts[1][:-1].strip()

        if not tool_name:
            raise ValueError("Tool name cannot be empty")

        if not args_str:
            return tool_name, {}

        try:
            args = self._parse_args(args_str)
        except Exception as e:
            raise ValueError(f"Error parsing arguments: {e}")

        return tool_name, args

    def _parse_args(self, args_str: str) -> Dict[str, Any]:
        """
        Parses the arguments string.
        This is a simplified parser and may not handle all edge cases.
        """
        args = {}
        # This is a simple parser. A more robust solution might be needed.
        # For now, we can wrap the args in a json-like structure and parse
        temp_json = f"{{{args_str}}}"
        try:
            # Replace single quotes with double quotes for JSON compatibility
            temp_json = temp_json.replace("'", '"')
            args = json.loads(temp_json)
        except json.JSONDecodeError:
            # Fallback for unquoted strings, e.g. document_name=my_doc
            # This is a very basic parser and has limitations.
            try:
                args = {}
                for arg in args_str.split(","):
                    key, value = arg.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    # Try to convert to int or bool
                    if value.isdigit():
                        value = int(value)
                    elif value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False
                    else:
                        # Remove quotes if present
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        elif value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]
                    args[key] = value
            except ValueError:
                raise ValueError(
                    "Could not parse arguments. Ensure they are in key=value format."
                )
        return args
