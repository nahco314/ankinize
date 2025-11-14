import enum
import re
from typing import Type, Any, Literal, Annotated

import json_repair
from agents import AgentOutputSchemaBase, ModelBehaviorError
from agents.strict_schema import ensure_strict_json_schema
from pydantic import BaseModel, TypeAdapter, Field


class SectionHeading(BaseModel):
    kind: Literal["section_heading"]
    content: str


class GroupHeading(BaseModel):
    kind: Literal["group_heading"]
    content: str


class PlainText(BaseModel):
    kind: Literal["plain_text"]
    content: str


class StartWord(BaseModel):
    kind: Literal["start_word"]
    word: str
    importance_level: int
    phonetic: str


class MeaningLine(BaseModel):
    kind: Literal["meaning_line"]
    content: str


class CheckProblem(BaseModel):
    kind: Literal["check_problem"]
    question: str
    answer: str


class StartReviewTest(BaseModel):
    kind: Literal["start_review_test"]
    pass


class ReviewTestSmallHeading(BaseModel):
    kind: Literal["review_test_small_heading"]
    content: str


class ReviewTestProblem(BaseModel):
    kind: Literal["review_test_problem"]
    question: str
    answer: str


type Event = Annotated[
    (
        SectionHeading
        | GroupHeading
        | PlainText
        | StartWord
        | MeaningLine
        | CheckProblem
        | StartReviewTest
        | ReviewTestSmallHeading
        | ReviewTestProblem
    ),
    Field(discriminator="kind"),
]


class Result(BaseModel):
    events: list[Event]


class MetaData(BaseModel):
    page: str


class FinalResult(BaseModel):
    result: Result
    metadata: MetaData


class StripFenceSchema(AgentOutputSchemaBase):
    def __init__(self, model: Type[BaseModel], strict_json_schema: bool = True):
        self.model = model
        self.ta = TypeAdapter(model)
        self._schema = self.ta.json_schema()
        self._strict = strict_json_schema
        if self._strict:
            self._schema = ensure_strict_json_schema(self._schema)

    def is_plain_text(self) -> bool:
        return False

    def name(self) -> str:
        return self.model.__name__

    def json_schema(self) -> dict[str, Any]:
        return self._schema

    def is_strict_json_schema(self) -> bool:
        return self._strict

    def validate_json(self, json_str: str) -> Any:
        s = json_str.strip()
        m = re.match(r"^\s*```(?:json|JSON)?\s*\n(.*?)\n?\s*```$", s, re.DOTALL)
        if m:
            s = m.group(1).strip()

        s = json_repair.repair_json(s)

        try:
            return self.ta.validate_json(s)  # ← ここで Pydantic 検証
        except Exception as e:
            print(json_str)
            print("----------------------------------------")
            print(s)
            raise ModelBehaviorError(
                f"Invalid JSON for {self.model.__name__}: {e}"
            ) from e


class Word(BaseModel):
    word: str
    importance_level: int
    phonetic: str
    meanings: list[str]


class Group(BaseModel):
    title: str
    lead: list[str]
    words: list[Word]
    check_problems: list[tuple[str, str]]
