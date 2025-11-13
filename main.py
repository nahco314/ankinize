import asyncio
import io
import json
import os
from functools import partial
from itertools import batched

import PIL
import openai
from openai import AsyncOpenAI
from outlines import from_openai
from outlines.inputs import Chat, Image
from pydantic import BaseModel
from pathlib import Path

from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    enable_verbose_stdout_logging,
    ModelSettings,
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
以下は、日本の大学受験用英単語帳である「鉄緑会　東大英単語英熟語　鉄壁」の数ページの画像です。
この英単語帳をデータ化するために、与えた全ページについて順に、構成要素をOCRし、後述する方法で本の角構成要素をイベントとして扱い、本の内容を厳密に上から順番にイベント列として出力してください。
具体的には、LLMを用いて単語帳に含まれる文章や、単語、見出しなどの各構成要素を「イベント」として扱いやすいデータ列に起こし、それらをPythonを用いて各ページごとに結合して、JSON等に構造化します。

各イベントの種別は以下の通りです。

- 大きな灰色の長方形で表される、「SECTION #1 「重要な・ささいな」」のように書いてある、セクションの大見出し
  - kind: section_heading
  - content: 見出しの内容をすべてそのまま含めてください。"SECTION #1"の部分や、「」の鉤括弧も原文ママで含めてください。
- 比較的小さな灰色の長方形で示される、「単語のブロックの前にある、その単語たちがどういうくくりなのか」という見出し。例えば「〜〜の類義語」みたいなやつ
  - kind: group_heading
  - content: 見出しの内容をすべてそのまま含めてください。
- 単語ではなく、後述する「Check!」にも「Review Test」にも含まれない、ただの本としての本文のテキスト
  - kind: plain_text
  - content: 本文の内容をすべてそのまま含めてください。文頭の●とかの記号なども含めてください。
- 新たな単語の開始(単語帳部分の左側に示される)
  - kind: start_word
  - word: 単語そのものをそのまま含めてください。
  - importance_level: 0, 1, または 2。単語の左側に赤いアスタリスク(*)が1つまたは2つ書いている場合があります。このときは、importance_levelを1または2にしてください。アスタリスクがない場合は0にしてください。
  - phonetic: 発音記号をそのまま含めてください。[]で囲まれていますが、これもphoneticに含めてください。
  - 単語の左に四角がありますが、これはチェック用の都合で配置されているものなので、OCRには含まないでください。
  - 同様に、単語番号も載っていますが、これもOCRには含まないでください。
  - **1つの単語について、複数の語義や説明、例文などが存在し、それぞれがmeaning_lineとなるため、1つのstart_wordは基本複数のmeaning_lineを含みます。**
- 語義や説明、例文、類義語など、単語の右側に書かれている部分の1行
  - kind: meaning_line
  - content: 右側部分の1行をそのまま含めてください。
  - 独自の形式で構造化されていますが、これらは記号などを可能な限り忠実にそのまま文字起こししてください。
    - 形容詞や名詞であることを表す、四角の中に「形」、みたいな記号は、四角の中の1文字を"[形]"というように[]で囲んで表示してください
      - 名詞: [名]
      - 自動詞: [自]
      - 他動詞: [他]
      - 句動詞: [句]
      - 形容詞: [形]
      - 副詞: [副]
    - ★や▶、□、あるいは①②などの記号はそのまま含めてください。
  - 「1行」の意味については、論理的な1行を指します。組版の都合で改行されているに過ぎない1部分は1つのmeaning_lineとして扱います。
  - 例文などに関しては、英文と日本語訳でそれぞれ別のmeaning_lineとしてください。
  - 赤い文字で表示されている、強調されている重要な語義や、強調部分は、<red></red>タグで囲んでください。
- 単語ブロックの最後に現れる、「Check!」のような軽い確認問題
  - kind: check_problem
  - question: 問題文をそのまま含めてください。
  - answer: 解答文をそのまま含めてください。
    - 解答は右端にかなり寄っています
  - 1つの「Check!」欄には複数の問題が含まれていることがありますが、その場合はすべてを1つ1つ別のcheck_problemとして扱ってください。
- 赤みがかった背景の、「Review Test」という見出し
  - kind: start_review_test
  - このイベントには特に他のフィールドは含まれません。
- Review Test内の小見出し。赤い丸で始まる、「Yes or No?」とか「Multiple Choices」みたいな、そのあとの問題群のリード文となる小見出し
  - kind: review_test_small_heading
  - content: 小見出しの内容をすべてそのまま含めてください。
- Review Test内の問題
  - kind: review_test_problem
  - question: 問題文をそのまま含めてください。
  - answer: 解答文をそのまま含めてください。

- 赤い文字で表示されている、強調されている重要な語義や、強調部分は、<red></red>タグで囲んでください。
- 与えられるページは断片的であるため、前のページで定義されたstart_wordに属するmeaning_lineからページが始まる場合などもあります。その場合は適切にmeaning_lineから始めてください。
- start_wordに属するmeaning_lineは、そのstart_wordのちょうど正確に真右から始まります。それより上のmeaning_lineは、更に前のstart_wordに属するものなので、「前のstart_wordに属するmeaning_lineを列挙する」→「新しいstart_wordを宣言」→「新しいstart_wordに属するmeaning_lineを列挙する」と適切に処理してください。
  - 特に、ページの一番最初が唐突に前ページの単語に属するmeaning_lineから始まる場合、極めてこれらをページ内の最初の単語に属するものと混同しやすいです。垂直方向の位置関係を極めてよく観察し、厳密に上に位置するイベントから順に列挙してください。
- ページの一番最初がreview_test_problemなどから始まる場合、これを単語などと混同してしまうかもしれません。「□12 To impose is to force somthing on others. ..................... Yes」みたいな、問題文→........→答え、みたいなやつはreview_test_problemです。
- **importance_levelは非常に誤りが多いです。極めて慎重に判定してください**

- **数ページ与えるので、全ページについてのイベント列を順に結合したものを出力してください**

- 文字種の指定:
  - 波線は、英文中では半角の「~」、日本語文中では全角の「〜」を使用
  - セミコロンは半角。複数の語義をセミコロンで繋ぐときは、半角セミコロン+半角スペース、すなわち「1つ; 1点; 1件」のようにする
- **出力されたJSONをそのままパースして使用するため、構文エラーやSchemaへの不適合がない、正確なJSONを出力するよう努めてください。**

---

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

    model = from_openai(
        openai.OpenAI(
            api_key=os.getenv("CLAUDE_API_KEY"),
            base_url="https://api.anthropic.com/v1/",
        ),
        "claude-opus-4-1-20250805",
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


def process(idx, name, p_range: list[int]):
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
            model="anthropic/claude-sonnet-4.5",
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
        metadata=MetaData(page=f"{' '.join(map(str, list(p_range)))}"),
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

    batchs = list(enumerate(batched(range(22, 22 + 712), 5)))

    for i, part in batchs[:1]:
        process(i, f"i", list(part))


if __name__ == "__main__":
    main()
