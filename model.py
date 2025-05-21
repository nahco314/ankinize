import enum

from pydantic import BaseModel


class SubProblemType(enum.Enum):
    verb = "動詞形"
    noun = "名詞形"
    adjective = "形容詞形"
    adverb = "副詞形"
    synonym = "同義語"
    antonym = "反義語"
    synonym_idiom = "同義語イディオム"
    accent = "アクセント"
    pronunciation = "発音"
    question = "Q"


class SubProblem(BaseModel):
    # type: SubProblemType
    question: str
    answer: str


class NoteType(enum.Enum):
    tagi = "多義"
    gohou = "語法"


class Word(BaseModel):
    word: str
    answer: str
    description: str
    sub_problems: list[SubProblem]
    notes: list[NoteType]
    phonetic: str


class MinimalPhrase(BaseModel):
    english: str
    japanese: str


class TagigoMinimalPhrase(BaseModel):
    headword: str
    phrases: list[MinimalPhrase]
    description: str
    phonetic: str


class Result(BaseModel):
    content: list[Word]


class TagigoResult(BaseModel):
    content: list[TagigoMinimalPhrase]


class MetaData(BaseModel):
    input_image_name: str


class FinalResult(BaseModel):
    result: Result | TagigoResult
    metadata: MetaData
