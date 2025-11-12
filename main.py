import asyncio
import json
import os
from functools import partial
from itertools import batched

from openai import AsyncOpenAI
from pydantic import BaseModel
from pathlib import Path

from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    enable_verbose_stdout_logging, ModelSettings,
)

from cloud_vision import cloud_vision_ocr
from model import Result, MetaData, FinalResult, StripFenceSchema
from utils import process_image, retrynize


# enable_verbose_stdout_logging()


class Context(BaseModel):
    paths: list[Path]

    json_schema: str


minimal_phrase_ranges = [
    range(66, 105),
    range(151, 198),
    range(224, 311),
]

normal_format_text = """
この英単語帳の仕様と、出力形式に関する注意は以下の通りです。
- 一般的な注意として、画像に存在する文字を、忠実に、一字一句同じように出力してください。
- 単語ごとの構成はこうなっています:
  - 左側: 英単語
  - 左下側: 発音
  - 右側: 日本語での意味、説明/補足、例文、類義語など
- 左側に配置されている英単語をそのままwordに含めてください。
  - 単語の左に四角がありますが、これはチェック用の都合で配置されているものなので、OCRには含まないでください。
  - 同様に、単語番号も載っていますが、これもOCRには含まないでください。
- 単語の下に、発音記号が書かれています。これをphoneticフィールドに含めてください。[]で囲まれていますが、これもphoneticに含めてください。
- 英単語の右側に赤いアスタリスク(*)が1つまたは2つ書いている場合があります。このときは、importance_levelを1または2にしてください。アスタリスクがない場合は0にしてください。
- 右側には、日本語での意味、説明/補足、例文、類義語などが含まれています。これをrightに含めてください。
- 右側部分は、独自の形式で構造化されていますが、これらは記号などを可能な限り忠実にそのまま文字起こししてください。
  - 形容詞や名詞であることを表す、四角の中に「形」、みたいな記号は、四角の中の1文字を"[形]"というように[]で囲んで表示してください
    - 名詞: [名]
    - 自動詞: [自]
    - 他動詞: [他]
    - 句動詞: [句]
    - 形容詞: [形]
    - 副詞: [副]
  - ★や▶、□、あるいは①②などの記号はそのまま含めてください。
  - 改行等は<br>を用いて、かつ原文の通り改行されるように行ってください。
- 赤い文字で表示されている、強調されている重要な語義や、強調部分は、<red></red>タグで囲んでください。
- 文字種の指定:
  - 波線は、英文中では半角の「~」、日本語文中では全角の「〜」を使用
  - セミコロンは半角。複数の語義をセミコロンで繋ぐときは、半角セミコロン+半角スペース、すなわち「1つ; 1点; 1件」のようにする
"""


# @retrynize
def process_normal(ctx: Context, ocr_agent) -> Result:
    llm_ocr_text = f"""
以下は、日本の大学受験用英熟語帳である「鉄緑会　東大英単語英熟語　鉄壁」の1セクションの画像です。
この英熟語帳をデータ化するために構造化して、指定されたスキーマのJSONとして出力してください。

- 比較的小さな灰色の長方形で示される、「単語のブロックの前にある、その単語たちがどういうくくりなのか、という見出し」を「title」と呼びます。例えば「〜〜の類義語」みたいなやつです。
  - **セクションとくくりは違います。** 「SECTION #1 「重要な・ささいな」」のように書いてある大見出しがセクションで、それより小さな区分けがくくりです。この、より小さな「くくり」の方を重視し、titleにもこちらを入れてください。
- また、くくりの表示の前に少し書いてある数行の、くくりについての説明、みたいな部分を「lead」と呼びます。「次は〜〜の類義語です。日本語では〜〜という〜〜ですが、...」みたいなやつ。
- また、そのくくりの単語が全部終わったあとに、「Check!」みたいな軽い数問がついてることがあります。これらはquestionとanswerからなるJSONオブジェクトのリストとして、check_questionsに格納してください。
  - 後述しますが、単語の中で赤く強調されている部分は<red></red>タグで囲みますが、Check!の部分はすべて赤いので、これは別に強調ではないので、<red>で囲まないでください。
- 各単語について、後述するフォーマットにしたがって、その単語自身(右側に書いているやつ)、発音、左側の説明・意味などをすべて文字起こしし、適切なJSONオブジェクトにまとめてください。
- 各くくりのブロックについて、{{"title": くくりのtitle, "lead": リード文, "check_questions": 前述のcheck_questionのリスト, "words": [このくくりに属する、後述する形式の単語のオブジェクトのリスト]}}という形式のJSONオブジェクトにまとめてください。

---

{normal_format_text}

JSON schema:
{ctx.json_schema}
"""

    images = []

    for p in ctx.paths:
        images.append(
            {
                "type": "input_image",
                "detail": "high",
                "image_url": process_image(p),
            }
        )

    llm_ocr_result = Runner.run_sync(
        ocr_agent,
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": llm_ocr_text,
                    },
                    *images,
                ],
            },
        ],
    )
    # llm_ocr_json = llm_ocr_result.final_output.model_dump_json()

    return llm_ocr_result.final_output


def process(idx, name, p_range: range):
    input_base = Path(f"inputs-teppeki")
    output_base = Path(f"outputs-teppeki")

    json_schema = json.dumps(Result.model_json_schema())

    external_client = AsyncOpenAI(
        api_key=os.getenv("OPEN_ROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    ocr_agent = Agent(
        name="Wordbook OCR",
        output_type=StripFenceSchema(Result),
        model=OpenAIChatCompletionsModel(
            model="anthropic/claude-opus-4.1",
            openai_client=external_client,
        ),
        instructions="指示に従い、OCR/構造化を行ってください。なお、出力には結果のみを含め、それ以外のあらゆる情報や補足、コードブロックの記号などは一切含まないでください。",
    )

    paths = [input_base / f"{i}.png" for i in p_range]

    ctx = Context(
        paths=paths,
        json_schema=json_schema,
    )

    res = process_normal(ctx, ocr_agent)

    final_res = FinalResult(
        result=res,
        metadata=MetaData(
            section_idx=idx,
            section=name,
        ),
    )

    with open(output_base / f"{idx}.json", "w") as f:
        f.write(final_res.model_dump_json())

    print(f"Processed {idx}: {name}")


def main():
    sections_raw = [
        ["SECTION #1", "重要な・ささいな", 1],
        ["SECTION #2", "特徴・明確さ・点", 13],
        ["SECTION #3", "構造・構成", 24],
        ["SECTION #4", "動詞をイメージする・1", 38],
        ["SECTION #5", "制約・強制・禁止", 53],
        ["SECTION #6", "阻害・除去・供給・促進", 65],
        ["SECTION #7", "目的・実行・達成", 77],
        ["SECTION #8", "時間", 94],
        ["SECTION #9", "金・経済", 107],
        ["SECTION #10", "場所・領域・範囲", 119],
        ["SECTION #11", "発生・繁栄・衰退・消滅", 133],
        ["SECTION #12", "多義語・1", 142],
        ["SECTION #13", "基本単語の確認", 159],
        ["SECTION #14", "関係・対立・一致", 179],
        ["SECTION #15", "言語・文学", 193],
        ["SECTION #16", "調査・研究", 208],
        ["SECTION #17", "議論・主張・要求", 217],
        ["SECTION #18", "語源から覚える", 230],
        ["SECTION #19", "力関係", 244],
        ["SECTION #20", "知覚・感覚・感情", 259],
        ["SECTION #21", "善悪・犯罪", 275],
        ["SECTION #22", "数・量", 287],
        ["SECTION #23", "思考・認識・知", 301],
        ["SECTION #24", "人・人生", 315],
        ["SECTION #25", "人間関係", 327],
        ["SECTION #26", "価値・基準・選択・出来事・参加", 340],
        ["SECTION #27", "政治", 354],
        ["SECTION #28", "産業", 365],
        ["SECTION #29", "医学・化学", 380],
        ["SECTION #30", "宗教・民族・慣習", 392],
        ["SECTION #31", "自然・環境", 404],
        ["SECTION #32", "短い単語", 417],
        ["SECTION #33", "傾向・可能性・反応", 431],
        ["SECTION #34", "衣食住・日常", 443],
        ["SECTION #35", "程度・割合", 453],
        ["SECTION #36", "動詞をイメージする・2", 466],
        ["SECTION #37", "基本動詞を用いた熟語表現・1", 480],
        ["SECTION #38", "熟語表現・2", 493],
        ["SECTION #39", "熟語表現・3", 505],
        ["SECTION #40", "心・身体", 518],
        ["SECTION #41", "コロケーションで覚える形容詞", 532],
        ["SECTION #42", "カタカナ英語", 547],
        ["SECTION #43", "教育・テクノロジー", 563],
        ["SECTION #44", "多義語・2", 575],
        ["SECTION #45", "歴史・軍事", 588],
        ["SECTION #46", "接続詞・副詞・前置詞", 598],
        ["SECTION #47", "難単語・1", 609],
        ["SECTION #48", "難単語・2", 619],
        ["SECTION #49", "難単語・3", 629],
        ["SECTION #50", "難単語・4", 639],
        ["null", "null", 653],
    ]

    sections = []
    for i in range(len(sections_raw) - 1):
        start = sections_raw[i][2]
        end = sections_raw[i + 1][2]
        sections.append(
            (i, sections_raw[i][0], sections_raw[i][1], range(start + 21, end + 21))
        )

    process(0, f"{sections[0][1]} {sections[0][2]}", sections[0][3])


if __name__ == "__main__":
    main()
