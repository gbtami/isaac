"""Render ACP form elicitation in the terminal and confirm the answers."""

import math
import re
from collections.abc import Callable

from acp import schema

from isaac_client.display import Terminal

FormMode = schema.ElicitationFormRequestMode | schema.ElicitationFormSessionMode
FormProperty = (
    schema.ElicitationStringPropertySchema
    | schema.ElicitationBooleanPropertySchema
    | schema.ElicitationIntegerPropertySchema
    | schema.ElicitationNumberPropertySchema
    | schema.ElicitationMultiSelectPropertySchema
)
FormValue = str | bool | int | float | list[str]


def options_for(field: FormProperty) -> list[tuple[str, str]]:
    if isinstance(field, schema.ElicitationStringPropertySchema):
        if field.one_of is not None:
            return [(item.const, item.title or item.const) for item in field.one_of]
        return [(value, value) for value in field.enum or []]
    if isinstance(field, schema.ElicitationMultiSelectPropertySchema):
        if isinstance(field.items, schema.UntitledMultiSelectItems):
            return [(value, value) for value in field.items.enum]
        return [(item.const, item.title or item.const) for item in field.items.any_of]
    if isinstance(field, schema.ElicitationBooleanPropertySchema):
        return [("true", "Yes"), ("false", "No")]
    return []


def default_input(value: FormValue | None, choices: list[tuple[str, str]]) -> str:
    if value is None:
        return ""
    if not choices:
        return str(value)
    values = value if isinstance(value, list) else [str(value).lower() if isinstance(value, bool) else str(value)]
    return ",".join(str(i) for i, (key, _) in enumerate(choices, 1) if key in values)


def parse_input(raw: str, field: FormProperty, choices: list[tuple[str, str]]) -> FormValue:
    if choices:
        indices = [int(part.strip()) for part in raw.split(",")]
        if len(set(indices)) != len(indices) or any(index < 1 or index > len(choices) for index in indices):
            raise ValueError("Select distinct numbers from the list.")
        values = [choices[index - 1][0] for index in indices]
        if isinstance(field, schema.ElicitationMultiSelectPropertySchema):
            maximum = field.max_items if field.max_items is not None else len(choices)
            if not (field.min_items or 0) <= len(values) <= maximum:
                raise ValueError("Choose the requested number of options.")
            return values
        if len(values) != 1:
            raise ValueError("Select one option.")
        return values[0] == "true" if isinstance(field, schema.ElicitationBooleanPropertySchema) else values[0]
    if isinstance(field, schema.ElicitationIntegerPropertySchema | schema.ElicitationNumberPropertySchema):
        value = int(raw) if isinstance(field, schema.ElicitationIntegerPropertySchema) else float(raw)
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Enter a finite number.")
        if (field.minimum is not None and value < field.minimum) or (
            field.maximum is not None and value > field.maximum
        ):
            raise ValueError("Enter a number within the requested range.")
        return value
    if isinstance(field, schema.ElicitationStringPropertySchema):
        if len(raw) < (field.min_length or 0) or (field.max_length is not None and len(raw) > field.max_length):
            raise ValueError("The answer is outside the requested length.")
        if field.pattern is not None and not re.search(field.pattern, raw):
            raise ValueError("The answer does not match the requested pattern.")
        return raw
    raise ValueError("No choices were provided for this field.")


async def answer_form(
    terminal: Terminal,
    agent: str,
    message: str,
    form: schema.ElicitationSchema,
    toolbar: Callable[[], str],
) -> schema.CreateElicitationResponse:
    fields = form.properties or {}
    required = set(form.required or [])
    if required - fields.keys():
        return schema.DeclineElicitationResponse(action="decline")
    answers: dict[str, FormValue | None] = {}
    async with terminal.menu():
        terminal.write(f"{agent} asks: {message}", style="bold")
        if form.description:
            terminal.write(form.description)
        try:
            while True:
                for key, field in fields.items():
                    title = field.title or key
                    if field.description:
                        terminal.write(field.description)
                    choices = options_for(field)
                    for index, (_, label) in enumerate(choices, 1):
                        terminal.write(f"  {index}. {label}")
                    hint = (
                        ", numbers separated by commas"
                        if isinstance(field, schema.ElicitationMultiSelectPropertySchema)
                        else ""
                    )
                    optional = " (optional)" if key not in required else ""
                    label = f"{title}{optional}{hint} > "
                    draft = default_input(answers.get(key, field.default), choices)
                    while True:
                        raw = await terminal.read(label, toolbar=toolbar, default=draft)
                        if not raw.strip() and key not in required:
                            answers[key] = None
                            break
                        try:
                            if not raw.strip():
                                raise ValueError("Enter an answer for this required field.")
                            answers[key] = parse_input(raw, field, choices)
                        except ValueError as exc:
                            terminal.write(str(exc), style="yellow")
                            draft = raw
                        else:
                            break

                terminal.write("Review answers", style="bold")
                supplied = {key: value for key, value in answers.items() if value is not None}
                for key, value in supplied.items():
                    terminal.write(f"  {fields[key].title or key}: {value}")
                choice = (
                    (await terminal.read("Send [y], edit [e], or decline [Enter] > ", toolbar=toolbar)).strip().lower()
                )
                if choice == "y":
                    return schema.AcceptElicitationResponse(action="accept", content=supplied)
                if choice != "e":
                    return schema.DeclineElicitationResponse(action="decline")
        except re.error:
            terminal.write("The form supplied an invalid pattern.", style="yellow")
            return schema.DeclineElicitationResponse(action="decline")
        except (EOFError, KeyboardInterrupt):
            return schema.CancelElicitationResponse(action="cancel")
