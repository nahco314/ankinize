import enum
import re
from typing import Type, Any

from agents import AgentOutputSchemaBase, ModelBehaviorError
from agents.strict_schema import ensure_strict_json_schema
from pydantic import BaseModel, TypeAdapter


class Word(BaseModel):
    word: str
    right: str
    importance_level: int
    phonetic: str


class CheckQuestion(BaseModel):
    question: str
    answer: str


class WordGroup(BaseModel):
    title: str
    lead: str
    check_questions: list[CheckQuestion]
    words: list[Word]


class Result(BaseModel):
    content: list[WordGroup]


class MetaData(BaseModel):
    section_idx: int
    section: str


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
        try:
            return self.ta.validate_json(s)  # ← ここで Pydantic 検証
        except Exception as e:
            print(json_str)
            print("----------------------------------------")
            print(s)
            raise ModelBehaviorError(
                f"Invalid JSON for {self.model.__name__}: {e}"
            ) from e
