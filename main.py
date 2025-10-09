import asyncio
import json
import os
from functools import partial
from itertools import batched

from openai import AsyncOpenAI
from pydantic import BaseModel
from pathlib import Path

from agents import Agent, Runner, OpenAIChatCompletionsModel, enable_verbose_stdout_logging

from cloud_vision import cloud_vision_ocr
from model import Result, MetaData, FinalResult, StripFenceSchema
from utils import process_image, retrynize


# enable_verbose_stdout_logging()


class Context(BaseModel):
    name: str
    normal_base: Path

    mono_ocr: str
    red_ocr: str

    json_schema: str


minimal_phrase_ranges = [
    range(66, 105),
    range(151, 198),
    range(224, 311),
]

all_minimal_phrase_nums = []
for r in minimal_phrase_ranges:
    all_minimal_phrase_nums.extend(r)


normal_format_text = """
この熟語帳の仕様と、出力形式に関する注意は以下の通りです。
- この熟語帳は、ページ上部に章タイトルやQRコード、ページ下部にフッターなどがあり、ノイズとして文字が紛れ込む場合がありますが、無視してください。
- 一般的な注意として、画像に存在する文字を、忠実に、一字一句同じように出力してください。
  - 画像が潰れていて/画質が悪くて識別不能な箇所は、文脈から判断してください。
- 単語ごとの構成はこうなっています:
  - 左上側: 英熟語
  - 右上側: 日本語での意味
  - 下側: 例文とその日本語訳
  - 最下部: 追加情報など(これはoptionalであり、ない場合もある)
- 上半分の左側に大きく配置されている、英熟語をそのままwordに含めてください。
- 英熟語の右上に、「口語」と書かれた吹き出しがある場合があります。このときは、出力のcasualをtrueにしてください。
- 上半分の右側(英熟語のちょうど右)には、日本語の意味が含まれています。これをanswerに含めてください。
- 英熟語と日本語訳が書かれている段の直下に、英語での例文があります。その後、続けて()で囲まれた日本語訳があります。それぞれをexample_en, example_jaに含めてください。
- 例文のうち、見出し語の部分は<red></red>タグで囲んで強調してください。英語側は、見出し語の表現が使われている箇所を、日本語側ではそれに対応する箇所をredで囲みます。
  - 例えば、「be late for 〜」という見出し語なら、
  - example_en: Don't <red>be late for</red> your appointment.
  - example_ja: 予定に<red>遅れ</red>ないようにしてください。
- さらにその下に、追加情報が書かれている場合があります。これらはすべてadditionalフィールドに含めてください。例えば以下のようなものがあります:
  - 対義語: 「⇔ in time (for 〜)」
  - 同義語: 「≒ at times」
  - 参照: 「cf. be afraid of ~ 「〜を恐れている」」
    - 参照する表現の日本語訳は、本ではそのまま繋げられていますが、出力では「」で囲んでください。
    - cf のあとにはピリオドをつけてください
  - その他いろいろなバリエーションがあります。例文などを含んでいるかなり長めのパターンもあります
    - additional の中において cf で参照した単語などの長い例文においても、その単語の部分は<red></red>で強調してください。表現のみの場合はいりません
- 文字種の指定:
  - 波線は、英文中では半角の「~」、日本語文中では全角の「〜」を使用
  - セミコロンは半角。複数の語義をセミコロンで繋ぐときは、半角セミコロン+半角スペース、すなわち「1つ; 1点; 1件」のようにする
"""


@retrynize
async def process_normal(ctx: Context, ocr_agent, structure_agent) -> Result:
    llm_ocr_text = f"""
以下は、日本の大学受験用英熟語帳である「速読英熟語」の1ページの画像です。
この英熟語帳をデータ化するために、このページを構造化して、指定されたスキーマのJSONとして出力してください。
なお、画像は見切れている場合があります。文章や単語が途中で切れていたり、欠けていたりする場合、文脈から予測し、適切に補ってください。

{normal_format_text}

JSON schema:
{ctx.json_schema}
"""

    llm_ocr_result = await Runner.run(
        ocr_agent,
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": llm_ocr_text,
                    },
                    {
                        "type": "input_image",
                        "detail": "high",
                        "image_url": process_image(ctx.normal_base / ctx.name),
                    },
                ],
            },
        ],
    )
    llm_ocr_json = llm_ocr_result.final_output.model_dump_json()

    main_text = f"""
以下は、日本の大学受験用英熟語帳である「速読英熟語」の1ページの画像を、LLMと古典的な手法の2通りでOCRしたデータです。
この英熟語帳をデータ化するために、このページを構造化して、指定されたスキーマのJSONとして出力してください。

- LLMの出力は既に構造化されています。しかし、文字の同定の精度に問題があったり、表現などに問題があるので、そういった点を古典的手法の結果と比べた上で校正してください。修正すべき間違いの例:
  - additionalでのcfには、ピリオドを付ける(「cf. 〜〜」というふうに)
- 古典的な手法のOCRの出力は、構造化が行われていません。テキストの順番や位置関係が実際のものと入れ替わっていたりします。しかし、文字自体の同定の精度は非常に高いため、単語や文字が実際に何であるかはLLMよりも信頼できます。

{normal_format_text}

LLM data:
{llm_ocr_json}

OCR data:
{ctx.mono_ocr}

JSON schema:
{ctx.json_schema}
"""

    result = await Runner.run(
        structure_agent,
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": main_text,
                    },
                ],
            },
        ],
    )

    return result.final_output


@retrynize
async def cloud_vision_ocr_async(image_path: Path) -> str:
    """
    同期実装の cloud_vision_ocr をスレッドプールでラップして
    疑似的に非同期化する例。
    """
    loop = asyncio.get_event_loop()
    # run_in_executor(None, func, *args) でデフォルトのスレッドプールを使う
    return await loop.run_in_executor(None, partial(cloud_vision_ocr, image_path))


async def process(num: int):
    class_id = "2-normal"
    mono_base = Path(f"processed-{class_id}/mono")
    red_base = Path(f"processed-{class_id}/red")
    normal_base = Path(f"processed-{class_id}")
    output_base = Path(f"outputs-{class_id}")

    name = f"{num}.png"

    mono_ocr_task = asyncio.create_task(cloud_vision_ocr_async(mono_base / name))
    red_ocr_task = asyncio.create_task(cloud_vision_ocr_async(red_base / name))
    mono_ocr, red_ocr = await asyncio.gather(mono_ocr_task, red_ocr_task)

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

    structure_agent = Agent(
        name="Wordbook OCR",
        output_type=StripFenceSchema(Result),
        model=OpenAIChatCompletionsModel(
            model="anthropic/claude-opus-4.1",
            openai_client=external_client,
        ),
        instructions="指示に従い、OCR/構造化を行ってください。なお、出力には結果のみを含め、それ以外のあらゆる情報や補足、コードブロックの記号などは一切含まないでください。",
    )

    ctx = Context(
        name=name,
        normal_base=normal_base,
        mono_ocr=mono_ocr,
        red_ocr=red_ocr,
        json_schema=json_schema,
    )

    if (output_base / f"{num}-minimal.json").exists():
        print(f"pass {num}")
        return

    if not num in all_minimal_phrase_nums:
        task_normal = asyncio.create_task(
            process_normal(ctx, ocr_agent, structure_agent)
        )
        res = await task_normal

        final_res = FinalResult(
            result=res,
            metadata=MetaData(
                input_image_name=name,
            ),
        )

        with open(output_base / f"{num}.json", "w") as f:
            f.write(final_res.model_dump_json())

    print(f"Processed {num}")


async def main():
    batch_lst = list(batched(range(144), 80))
    tasks = [process(i) for i in batch_lst[1]]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
