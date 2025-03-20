import asyncio
import json
import os
from functools import partial
from itertools import batched

from openai import AsyncOpenAI
from pydantic import BaseModel
from pathlib import Path

from agents import Agent, Runner, OpenAIChatCompletionsModel

from cloud_vision import cloud_vision_ocr
from model import Result, MetaData, FinalResult
from utils import process_image, retrynize


class Context(BaseModel):
    name: str
    normal_base: Path

    mono_ocr: str
    red_ocr: str

    normal_json_schema: str


minimal_phrase_ranges = [
    range(140, 179),
    range(225, 272),
]

all_minimal_phrase_nums = []
for r in minimal_phrase_ranges:
    all_minimal_phrase_nums.extend(r)


normal_format_text = """
この単語帳の仕様と、出力形式に関する注意は以下の通りです。
- この単語帳は、ページ上部に「MINIMAL PHRASES」という、重要フレーズ集があります。今回はこれは扱わないため、完全に無視してOKです。
- 一般的な注意として、画像に存在する文字を、忠実に、一字一句同じように出力してください。
  - 画像が潰れていて/画質が悪くて識別不能な箇所は、文脈から判断してください。
- 単語ごとの構成はこうなっています:
  - 左側: 英単語、発音記号、単語にまつわる問題(sub_problem)
  - 右側: 日本語での意味、頻出する形、関連語、sub_problemの答えなど
- 左側に位置する、英単語をそのままwordに含めてください。
- 左側には、語に関する注意(notes)があることがあります。これらはnotesフィールドにリストとして含めてください。種類は以下の通りです。
  - 角丸線で囲んだ「多義」: 単語に複数の意味があることに注意
  - 角丸線で囲んだ「語法」: 語法や構文に注意
- 同じように、左側には、単語について注意する必要がある項目に関する問題が含まれることがあります。これらは、その右側に答えが位置しているため、その答えと併せてsub_problemsフィールドにリストとして含めてください。
  動詞形などを聞く問題は「動詞形は何ですか」などを問題文として、一般の問題(Q)はその問題文を問題文として、答えは普通に答えを含めてください。
  また、「同？2つ」などの形で、問題に関するさらなる補足が含まれる場合もあるため、それも問題分として含めてください。
  種類は以下の通りです。
  - 角丸線で囲んだ「動？」: 動詞形は何ですか
  - 角丸線で囲んだ「名？」: 名詞形は何ですか
  - 角丸線で囲んだ「形？」: 形容詞形は何ですか
  - 角丸線で囲んだ「副？」: 副詞形は何ですか
  - 角丸線で囲んだ「接？」: 接続詞形は何ですか
  - 角丸線で囲んだ「同？」: 同義語は何ですか
  - 角丸線で囲んだ「反？」: 反意語は何ですか
  - 角丸線で囲んだ「同熟？」: 同じ意味を表す熟語は何ですか
  - 角丸線で囲んだ「アク」: アクセントは何ですか
    - このタイプの単語は、右側に発音記号があります。発音記号は普通に出力に含めた上で、「アクセントは何ですか」が問題文、発音記号を答えとするsub_problemにしてください。
  - 角丸線で囲んだ「語法」: 語法や構文に注意
    - これは、「語法や構文に注意」を問題文として、その語法や構文を答えとするsub_problemにしてください。
  - Q: それ以外の一般の問題
- 右側には、日本語の意味が含まれています。多義語の場合などは、①②などでいくつかの意味が列挙されている場合がありますが、これらはすべてanswerに含めてください。
- 右側の特に重要な部分は赤い文字で書かれています。そのような部分は、<red></red>タグで囲んでください。
- 日本語での意味における、「動」や「名」などの漢字を四角で囲んだ記号は、「動詞形」などという意味です。出力には、`<動>`のように含めてください。
- 日本語での意味の説明において、存在する記号は以下の通りです。これらはそのままの記号で出力に含めてください(四角で囲んだ「動」などは、`<動>`のように含めてください。)。
  - 四角で囲んだ「動」: 動詞形
  - 四角で囲んだ「名」: 名詞形
  - 四角で囲んだ「形」: 形容詞形
  - 四角で囲んだ「副」: 副詞形
  - 四角で囲んだ「接」: 接続詞形
  - 四角で囲んだ「前」: 前置詞形
  - 四角で囲んだ「源」: 語源の説明
  - 四角で囲んだ「諺」: ことわざ
  - =: 同義語
  - ⇔: 反意語
  - ◇ (中空のひし形): 派生語・関連語
  - ◆ (中黒のひし形): 熟語・成句
  - cf.: 参照
  - <<米>>: 米国での使い方
  - <<英>>: 英国での使い方
- 単語の品詞を表す、漢字1文字の「動」や「形」は、前述した通り<動>や<形>としてください。
- 単語の日本語表現(複数ある場合もある)と、それに直接含まれる説明のみをanswerに、それ以外の説明をdescriptionに含めてください。
- 左側、あるいは稀に右側に位置する発音記号(アクセントも含む)をphoneticに含めてください。
"""


minimal_phrase_format_text = """
この単語帳の仕様と、出力形式に関する注意は以下の通りです。
- この単語帳は、ページ上部に「MINIMAL PHRASES」という、重要フレーズ集があります。今回はこれのみをOCRしてください。単語単体の部分は無視してください。
- 一般的な注意として、画像に存在する文字を、忠実に、一字一句同じように出力してください。
  - 画像が潰れていて/画質が悪くて識別不能な箇所は、文脈から判断してください。
- フレーズごとの構成はこうなっています:
  - 左側: 英語でのフレーズ、単語にまつわる問題(sub_problem)
  - 右側: 日本語での意味、関連語、sub_problemの答えなど
- 左右両方について、特に重要な部分や、問題として問いたい部分は赤い文字で書かれています。そのような部分は、<red></red>タグで囲んでください。
- 左側に位置するフレーズをまとめてwordに含めてください。
- 左側には、単語について注意する必要がある項目に関する問題が含まれることがあります。これらは、その右側に答えが位置しているため、その答えと併せてsub_problemsフィールドにリストとして含めてください。
  動詞形などを聞く問題は「動詞形は何ですか」などを問題文として、一般の問題(Q)はその問題文を問題文として、答えは普通に答えを含めてください。
  また、「同？2つ」などの形で、問題に関するさらなる補足が含まれる場合もあるため、それも問題分として含めてください。
  種類は以下の通りです。
  - 角丸線で囲んだ「動？」: 動詞形は何ですか
  - 角丸線で囲んだ「名？」: 名詞形は何ですか
  - 角丸線で囲んだ「形？」: 形容詞形は何ですか
  - 角丸線で囲んだ「副？」: 副詞形は何ですか
  - 角丸線で囲んだ「接？」: 接続詞形は何ですか
  - 角丸線で囲んだ「同？」: 同義語は何ですか
  - 角丸線で囲んだ「反？」: 反意語は何ですか
  - 角丸線で囲んだ「同熟？」: 同じ意味を表す熟語は何ですか
  - 角丸線で囲んだ「語法」: 語法や構文に注意
    - これは、「語法や構文に注意」を問題文として、その語法や構文を答えとするsub_problemにしてください。
  - Q: それ以外の一般の問題
- 右側には、日本語の意味が含まれています。
- 日本語での意味における、「動」や「名」などの漢字を四角で囲んだ記号は、「動詞形」などという意味です。出力には、`<動>`のように含めてください。
- 日本語での意味の説明において、存在する記号は以下の通りです。これらはそのままの記号で出力に含めてください(四角で囲んだ「動」などは、`<動>`のように含めてください。)。
  - 四角で囲んだ「動」: 動詞形
  - 四角で囲んだ「名」: 名詞形
  - 四角で囲んだ「形」: 形容詞形
  - 四角で囲んだ「副」: 副詞形
  - 四角で囲んだ「接」: 接続詞形
  - 四角で囲んだ「前」: 前置詞形
  - 四角で囲んだ「源」: 語源の説明
  - 四角で囲んだ「諺」: ことわざ
  - =: 同義語
  - ⇔: 反意語
  - ◇ (中空のひし形): 派生語・関連語
  - ◆ (中黒のひし形): 熟語・成句
  - cf.: 参照
  - <<米>>: 米国での使い方
  - <<英>>: 英国での使い方
- 単語の品詞を表す、漢字1文字の「動」や「形」は、前述した通り<動>や<形>としてください。
- 単語の日本語表現(複数ある場合もある)と、それに直接含まれる説明のみをanswerに、それ以外の説明をdescriptionに含めてください。
- 左側、あるいは稀に右側に位置する発音記号(アクセントも含む)をphoneticに含めてください。存在しない場合は空文字列で結構です。
"""


@retrynize
async def process_normal(ctx: Context, ocr_agent, structure_agent) -> Result:
    llm_ocr_text = f"""
以下は、日本の大学受験用英単語帳である「システム英単語」の1ページの画像です。
この英単語帳をデータ化するために、このページを構造化して、指定されたスキーマのJSONとして出力してください。

{normal_format_text}

JSON schema:
{ctx.normal_json_schema}
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
    llm_ocr_json = llm_ocr_result.final_output.model_dump_json(indent=2)

    main_text = f"""
以下は、日本の大学受験用英単語帳である「システム英単語」の1ページの画像を、LLMと古典的な手法の2通りでOCRしたデータです。
この英単語帳をデータ化するために、このページを構造化して、指定されたスキーマのJSONとして出力してください。
- LLMの出力は既に構造化されています。しかし、文字の同定の精度に問題があったり、表現などに問題があるので、そういった点を古典的手法の結果と比べた上で校正してください。修正すべき間違いの例:
  - 漢字の「名？」と「動」を読み間違えて、「名詞化は？」と「動詞化は？」を間違える
  - 「名詞化は？」などと記述しなければならないところを、「名？」などとのみ書いてしまう
  - <動>や<形>としなければならない場所を漢字1文字で動や形などとしている
  - ◇や◆とすべきところを、☆や★としている
    - ★は◆とは違う意味。◇や◆は関連語句の例示、★はワンポイントアドバイス的な説明の明示。
    - 基本的に語句の前には◇や◆を使う
  - <動>や<形>、◇や◆を抜かしてしまっている
  - sub_problemsにおいて、選択肢をanswerに入れてしまっている。選択肢や問題文全てをquestionに、答えとその解説をanswerに入れること
  - 「多義」のnoteがない単語に「多義」と付けてしまう。**単に単語の日本語訳が①②と複数あるだけでは「多義」は付かない**。実際に「多義」が(古典的手法で)OCRされているときだけ付けること。LLMはハルシネーションによって不要な「多義」をよく付けるので、古典的OCRとしっかり比較すること
  - sub_problemsでの「Q」のタイプの問題は「Q. 〇〇の〜〜は？」というフォーマットにする
  - sub_problemsでの答え部分にAみたいなのを付ける場合は、ピリオドを付ける(「A. 〜〜」というふうに)
  - **「」ではなく[]を使用してしまっている。発音記号の箇所以外は、基本的にかぎかっことして「」を使うこと**
- 古典的な手法のOCRの出力は、構造化が行われていません。テキストの順番や位置関係が実際のものと入れ替わっていたりします。しかし、文字自体の同定の精度は非常に高いため、単語や文字が実際に何であるかはLLMよりも信頼できます。
- 古典的なOCRとして、「赤い部分のみの画素を用いてOCRしたもの」も与えます。どこが赤いテキストになっているかの参考にしてください。

{normal_format_text}

LLM data:
{llm_ocr_json}

OCR data:
{ctx.mono_ocr}

OCR data (only red parts):
{ctx.red_ocr}

JSON schema:
{ctx.normal_json_schema}
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
async def process_minimal_phrase(ctx: Context, ocr_agent, structure_agent) -> Result:
    llm_ocr_text = f"""
以下は、日本の大学受験用英単語帳である「システム英単語」の1ページの画像です。
この英単語帳をデータ化するために、このページを構造化して、指定されたスキーマのJSONとして出力してください。

{minimal_phrase_format_text}

JSON schema:
{ctx.normal_json_schema}
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
    llm_ocr_json = llm_ocr_result.final_output.model_dump_json(indent=2)

    main_text = f"""
以下は、日本の大学受験用英単語帳である「システム英単語」の1ページの画像を、LLMと古典的な手法の2通りでOCRしたデータです。
この英単語帳をデータ化するために、このページを構造化して、指定されたスキーマのJSONとして出力してください。
- LLMの出力は既に構造化されています。しかし、文字の同定の精度に問題があったり、表現などに問題があるので、そういった点を古典的手法の結果と比べた上で校正してください。修正すべき間違いの例:
  - 漢字の「名？」と「動」を読み間違えて、「名詞化は？」と「動詞化は？」を間違える
  - 「名詞化は？」などと記述しなければならないところを、「名？」などとのみ書いてしまう
  - <動>や<形>としなければならない場所を漢字1文字で動や形などとしている
  - ◇や◆とすべきところを、☆や★としている
    - ★は◆とは違う意味。◇や◆は関連語句の例示、★はワンポイントアドバイス的な説明の明示。
    - 基本的に語句の前には◇や◆を使う
  - <動>や<形>、◇や◆を抜かしてしまっている
  - sub_problemsにおいて、選択肢をanswerに入れてしまっている。選択肢や問題文全てをquestionに、答えとその解説をanswerに入れること
  - 「多義」のnoteがない単語に「多義」と付けてしまう。**単に単語の日本語訳が①②と複数あるだけでは「多義」は付かない**。実際に「多義」が(古典的手法で)OCRされているときだけ付けること。LLMはハルシネーションによって不要な「多義」をよく付けるので、古典的OCRとしっかり比較すること
  - sub_problemsでの「Q」のタイプの問題は「Q. 〇〇の〜〜は？」というフォーマットにする
  - sub_problemsでの答え部分にAみたいなのを付ける場合は、ピリオドを付ける(「A. 〜〜」というふうに)
  - **「」ではなく[]を使用してしまっている。発音記号の箇所以外は、基本的にかぎかっことして「」を使うこと**
  - 英語のフレーズ(wordフィールド)に<red>で囲まれた部分が存在しない。すべてのフレーズには、赤い部分が存在するはずなので、それを見つけて<red>で囲むこと
- 古典的な手法のOCRの出力は、構造化が行われていません。テキストの順番や位置関係が実際のものと入れ替わっていたりします。しかし、文字自体の同定の精度は非常に高いため、単語や文字が実際に何であるかはLLMよりも信頼できます。

{minimal_phrase_format_text}

LLM data:
{llm_ocr_json}

OCR data:
{ctx.mono_ocr}

OCR data (only red parts):
{ctx.red_ocr}

JSON schema:
{ctx.normal_json_schema}
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
    class_id = "0-normal"
    mono_base = Path(f"processed-{class_id}/mono")
    red_base = Path(f"processed-{class_id}/red")
    normal_base = Path(f"processed-{class_id}")
    output_base = Path(f"outputs-{class_id}")

    name = f"{num}.png"

    mono_ocr_task = asyncio.create_task(cloud_vision_ocr_async(mono_base / name))
    red_ocr_task = asyncio.create_task(cloud_vision_ocr_async(red_base / name))
    mono_ocr, red_ocr = await asyncio.gather(mono_ocr_task, red_ocr_task)

    json_schema = json.dumps(Result.model_json_schema(), indent=2)

    external_client = AsyncOpenAI(
        api_key=os.getenv("OPEN_ROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    ocr_agent = Agent(
        name="Wordbook OCR",
        output_type=Result,
        model=OpenAIChatCompletionsModel(
            model="anthropic/claude-3.7-sonnet",
            openai_client=external_client,
        ),
        instructions="指示に従い、OCR/構造化を行ってください。なお、出力には結果のみを含め、それ以外のあらゆる情報や補足、コードブロックの記号などは一切含まないでください。",
    )

    structure_agent = Agent(
        name="Wordbook OCR",
        output_type=Result,
        model=OpenAIChatCompletionsModel(
            model="anthropic/claude-3.7-sonnet",
            openai_client=external_client,
        ),
        instructions="指示に従い、OCR/構造化を行ってください。なお、出力には結果のみを含め、それ以外のあらゆる情報や補足、コードブロックの記号などは一切含まないでください。",
    )

    ctx = Context(
        name=name,
        normal_base=normal_base,
        mono_ocr=mono_ocr,
        red_ocr=red_ocr,
        normal_json_schema=json_schema,
    )

    if not num in all_minimal_phrase_nums:
        task_normal = asyncio.create_task(process_normal(ctx, ocr_agent, structure_agent))
        task_minimal = asyncio.create_task(process_minimal_phrase(ctx, ocr_agent, structure_agent))
        res, res_minimal = await asyncio.gather(task_normal, task_minimal)

        final_res = FinalResult(
            result=res,
            metadata=MetaData(
                input_image_name=name,
            ),
        )
        final_res_minimal = FinalResult(
            result=res_minimal,
            metadata=MetaData(
                input_image_name=name,
            ),
        )

        with open(output_base / f"{num}.json", "w") as f:
            f.write(final_res.model_dump_json(indent=2))

        with open(output_base / f"{num}-minimal.json", "w") as f:
            f.write(final_res_minimal.model_dump_json(indent=2))

    else:
        res = await process_minimal_phrase(
            ctx,
            ocr_agent,
            structure_agent,
        )
        final_res = FinalResult(
            result=res,
            metadata=MetaData(
                input_image_name=name,
            ),
        )

        with open(output_base / f"{num}-minimal.json", "w") as f:
            f.write(final_res.model_dump_json(indent=2))

    print(f"Processed {num}")


async def main():
    # 272ページまでを10ページ刻みでバッチにする例
    batch_lst = list(batched(range(272), 50))
    # ここでは最初のバッチのみ処理する例
    tasks = [process(i) for i in batch_lst[5]]
    await asyncio.gather(*tasks)


if __name__ == '__main__':
    asyncio.run(main())
