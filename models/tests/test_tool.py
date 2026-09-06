from models.tool import FuncDef, ParametertDef


def test_function_schema_uses_chat_completions_shape():
    function = FuncDef(
        name="get_weather",
        description="Get the current weather.",
        parameters=[
            ParametertDef(
                name="city",
                value_type=str,
                description="City name.",
                enums=["Taipei", "Tokyo"],
            )
        ],
    )

    assert function.json_schema() == {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name.",
                        "enum": ["Taipei", "Tokyo"],
                    }
                },
                "required": ["city"],
            },
        },
    }
