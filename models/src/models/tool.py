from dataclasses import dataclass
from typing import Any


@dataclass
class ParametertDef:
    name: str
    value_type: type
    description: str
    enums: list[Any]
    required: bool = True


@dataclass
class FuncDef:
    name: str
    description: str
    parameters: list[ParametertDef]

    def get_type(self, value_type: type) -> str:
        if value_type is bool:
            return "boolean"
        if value_type is int or value_type is float:
            return "number"
        if value_type is str:
            return "string"
        if value_type is list or value_type is set:
            return "array"
        if value_type is dict:
            return "object"
        if value_type is type(None):
            return "null"

        raise TypeError("unsupported type: ", value_type)

    def json_schema(self) -> dict:
        func_def_template = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": {}},
            },
        }

        required_parameters: list[str] = []
        for p in self.parameters:
            if p.required:
                required_parameters.append(p.name)

            func_def_template["function"]["parameters"]["properties"][p.name] = {
                "type": self.get_type(p.value_type),
                "description": p.description,
            }

            if len(p.enums):
                func_def_template["function"]["parameters"]["properties"][p.name][
                    "enum"
                ] = p.enums

        if required_parameters:
            func_def_template["function"]["parameters"]["required"] = (
                required_parameters
            )

        return func_def_template
