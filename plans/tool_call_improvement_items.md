# Tool Call Improvement Items

## Scope

- Preserve the current learning-oriented agent loop.
- Add parameterized tool execution without trusting model-generated arguments.
- Keep tool execution limited to explicitly registered tools.

## P1 — Parse and pass tool arguments

Current behavior:

- `ToolCall.arguments` is collected from both streamed and non-streamed responses.
- `Agent` invokes a registered tool with only the tool-call ID, so tools cannot receive model-generated parameters.

Improvement:

- Parse `ToolCall.arguments` as JSON after the complete tool call has been assembled.
- Pass the parsed arguments to the registered callable separately from `tool_call_id`.
- Keep `tool_call_id` as message metadata rather than a tool-function argument.

Acceptance criteria:

- A tool with one or more parameters receives the complete parsed arguments.
- Multiple tool calls retain their own arguments and IDs.
- Invalid JSON produces a tool error or controlled agent error rather than executing the tool.
- Tests cover streamed argument fragments, non-streamed arguments, and multiple tool calls.

## P1 — Validate arguments before execution

Current behavior:

- Registered tools form an allowlist, but model-generated arguments are not parsed or validated.
- The current `get_current_time` tool has no parameters and no side effects, so this does not create an immediate security issue.

Improvement:

- Validate parsed arguments against the registered tool definition before invocation.
- Reject missing required fields, unexpected fields, invalid enum values, and incorrect value types.
- Do not use `eval`, dynamic imports, or model-provided function names outside the explicit registry.
- For future tools with side effects, keep authorization and confirmation policy outside the model-generated arguments.

Acceptance criteria:

- Invalid arguments never reach the registered callable.
- Unknown tool names never invoke arbitrary code.
- Tool validation failures are returned or raised in a controlled and testable form.
- Side-effecting tools can add explicit authorization checks before execution.

## Suggested order

1. Define a typed registered-tool interface.
2. Parse the completed `arguments` JSON string.
3. Validate it against `FuncDef.parameters`.
4. Invoke the allowlisted callable with validated arguments.
5. Add tests for malformed, unexpected, and valid arguments.

## Reference

- <https://developers.openai.com/api/reference/cli/resources/chat/subresources/completions>
