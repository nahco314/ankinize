import json
import re
from pathlib import Path
from model import (
    FinalResult,
    GroupHeading,
    Event,
    SectionHeading,
    Word,
    Group,
    PlainText,
    StartWord,
    MeaningLine,
    CheckProblem,
)


def parse_to_sections(events: list[Event]) -> list[tuple[SectionHeading, list[Event]]]:
    sections = []
    sec_buf = []

    for e in events:
        if isinstance(e, SectionHeading):
            sections.append(sec_buf)
            sec_buf = []
        sec_buf.append(e)

    sections.append(sec_buf)

    sections.pop(0)

    res = []
    for sec in sections:
        heading = sec[0]
        body = sec[1:]
        res.append((heading, body))

    return res


def parse_to_groups(events: list[Event]) -> list[Group]:
    groups = []
    group_buf = None
    next_texts_buf = []
    word_buf = None

    for e in events:
        match e:
            case PlainText(content=c):
                next_texts_buf.append(c)
            case GroupHeading(content=c):
                if group_buf is not None:
                    if word_buf is not None:
                        group_buf.words.append(word_buf)
                        word_buf = None
                    groups.append(group_buf)
                group_buf = Group(
                    lead=next_texts_buf, words=[], check_problems=[], title=c
                )
                next_texts_buf = []
            case StartWord(word=w, importance_level=il, phonetic=p):
                if word_buf is not None:
                    group_buf.words.append(word_buf)
                word_buf = Word(word=w, importance_level=il, phonetic=p, meanings=[])
            case MeaningLine(content=c):
                assert word_buf is not None
                word_buf.meanings.append(c)
            case CheckProblem(question=q, answer=a):
                group_buf.check_problems.append((q, a))
            case _:
                pass

    if word_buf is not None:
        group_buf.words.append(word_buf)

    groups.append(group_buf)

    return groups


def parse_to_groups_without_group_heading(
    events: list[Event], title: str
) -> list[Group]:
    cnt = 0
    groups = []
    group_buf = Group(lead=[], words=[], check_problems=[], title=f"{title}: {cnt}")
    word_buf = None

    for e in events:
        match e:
            case PlainText(content=c):
                if len(group_buf.words) > 0:
                    if word_buf is not None:
                        group_buf.words.append(word_buf)
                        word_buf = None
                    groups.append(group_buf)
                    cnt += 1
                    group_buf = Group(
                        lead=[], words=[], check_problems=[], title=f"{title}: {cnt}"
                    )
                group_buf.lead.append(c)
            case GroupHeading():
                assert False
            case StartWord(word=w, importance_level=il, phonetic=p):
                if word_buf is not None:
                    group_buf.words.append(word_buf)
                word_buf = Word(word=w, importance_level=il, phonetic=p, meanings=[])
            case MeaningLine(content=c):
                assert word_buf is not None
                word_buf.meanings.append(c)
            case CheckProblem(question=q, answer=a):
                group_buf.check_problems.append((q, a))
            case _:
                pass

    if word_buf is not None:
        group_buf.words.append(word_buf)

    groups.append(group_buf)

    return groups


def parse_events(events: list[Event]) -> list[tuple[SectionHeading, list[Group]]]:
    sections = parse_to_sections(events)
    res = []

    for heading, body in sections:
        cc = sum(isinstance(e, GroupHeading) for e in body)
        if cc != 0:
            groups = parse_to_groups(body)
        else:
            groups = parse_to_groups_without_group_heading(body, heading.content)
        res.append((heading, groups))

    return res


def preprocess_0(events: list[Event]) -> list[Event]:
    all_ = True
    any_ = False
    for e in events:
        match e:
            case StartWord(importance_level=i):
                if i == 2:
                    all_ = False
                if i == 1:
                    any_ = True
                # print(i)

    return events


def preprocess_1(events: list[Event]) -> list[Event]:
    res = []

    for e in events:
        match e:
            case PlainText(content=c) if c.startswith("* "):
                res.append(
                    StartWord(
                        kind="start_word", word=c[2:], importance_level=2, phonetic=""
                    )
                )
            case _:
                res.append(e)

    return res


def main():
    raw_base = Path("./raw-outputs-teppeki")

    events = []

    for i in range(131):
        print(i)

        raw_path = raw_base / f"{i}.json"
        content = raw_path.read_text(encoding="utf-8")
        final_res = FinalResult.model_validate_json(content)
        res = final_res.result

        # events = preprocess_0(res.events)

        events.extend(res.events)

    events = preprocess_1(events)

    print(len(events))

    res = parse_events(events)

    for _, gs in res:
        for g in gs:
            for word in g.words:
                for i in range(len(word.meanings)):
                    m = word.meanings[i]
                    m = re.sub("[‡]", "", m)
                    m = re.sub(" → p\.\d+", "", m)
                    # m = re.sub("\[.*?]", "", m)
                    word.meanings[i] = m


    with open("importance.json", "r") as f:
        importance_data = json.load(f)

    cnt = 0

    # 143
    # 135
    # 120

    for w_l, imp in importance_data:
        w = " ".join(w_l)
        w = w.replace("[", "")
        w = w.replace("]", "")
        w = w.replace("0", "o")
        w = w.replace("_", "")
        w = w.replace("(", "")
        w = w.replace(")", "")
        w = w.replace(":", "")
        if w == "rid of get":
            w = "get rid of"
        found = False
        for _, gs in res:
            for g in gs:
                for word in g.words:
                    if word.word == w:
                        found = True
                        word.importance_level = imp

        if not found:
            pass
            print("!!!", w)
            cnt += 1

    print(cnt)

    cc = 0
    for _, gs in res:
        for g in gs:
            for word in g.words:
                cc += 1

    print(cc)

    # return

    # for s, gs in res:
    #     print(f"# {s}")
    #
    #     for g in gs:
    #         print(f"## {g.title}")
    #         print()
    #         for l in g.lead:
    #             print(l)
    #         print()
    #
    #         for w in g.words:
    #             print(w.word)
    #             for m in w.meanings:
    #                 print(f"    {m}")

    data_to_save = []
    for s, gs in res:
        lst = []
        for g in gs:
            lst.append(g.model_dump())
        data_to_save.append(
            {
                "section_heading": s.content,
                "groups": lst,
            }
        )

    with open("res.json", "w") as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
