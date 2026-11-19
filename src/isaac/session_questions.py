"""Present Harness AskUser questions through standard ACP form elicitation."""

import acp
from acp import schema
from pydantic_ai_harness.ask_user import AskUser, AskUserAnswer, AskUserRequest, AskUserResponse, check_response
from pydantic_ai_harness.experimental.acp import ACPSession

from isaac.config import SessionSettings


def session_questions(session: ACPSession) -> AskUser[SessionSettings] | None:
    capabilities = session.client_capabilities
    if capabilities is None or capabilities.elicitation is None or capabilities.elicitation.form is None:
        return None

    async def answer(request: AskUserRequest) -> AskUserResponse:
        properties: dict[str, schema.ElicitationStringPropertySchema | schema.ElicitationMultiSelectPropertySchema] = {}
        for index, question in enumerate(request.questions):
            choices = [
                schema.EnumOption(
                    const=option.label,
                    title=f"{option.label} · {option.description}" if option.description else option.label,
                )
                for option in question.options
            ]
            if question.multi_select:
                properties[f"choice_{index}"] = schema.ElicitationMultiSelectPropertySchema(
                    type="array",
                    title=question.header,
                    description=question.question,
                    min_items=1,
                    items=schema.TitledMultiSelectItems(any_of=choices),
                )
            else:
                properties[f"choice_{index}"] = schema.ElicitationStringPropertySchema(
                    type="string", title=question.header, description=question.question, one_of=choices
                )
            properties[f"other_{index}"] = schema.ElicitationStringPropertySchema(
                type="string", title=f"Other answer for {question.header}"
            )

        try:
            result = await session.client.create_elicitation(
                message="Choose an option or enter an Other answer for each question.",
                mode=schema.ElicitationFormSessionMode(
                    session_id=session.session_id,
                    requested_schema=schema.ElicitationSchema(type="object", properties={**properties}),
                ),
            )
        except acp.RequestError:
            return AskUserResponse(cancelled=True)
        if not isinstance(result, schema.AcceptElicitationResponse) or not result.content:
            return AskUserResponse(cancelled=True)

        answers = []
        for index, question in enumerate(request.questions):
            choice = result.content.get(f"choice_{index}")
            other = result.content.get(f"other_{index}")
            if isinstance(other, str) and other.strip():
                answers.append(AskUserAnswer(header=question.header, custom_answer=other))
            elif isinstance(choice, str):
                answers.append(AskUserAnswer(header=question.header, selected=(choice,)))
            elif isinstance(choice, list) and all(isinstance(item, str) for item in choice):
                answers.append(AskUserAnswer(header=question.header, selected=tuple(choice)))
        response = AskUserResponse(answers=tuple(answers))
        try:
            check_response(request, response)
        except ValueError:
            return AskUserResponse(cancelled=True)
        return response

    return AskUser(answerer=answer)
