import json
import os

from openai import AsyncOpenAI
from pydantic import BaseModel
from pathlib import Path

from agents import Agent, Runner, OpenAIChatCompletionsModel

from cloud_vision import cloud_vision_ocr
from model import Word
from utils import process_image


class Result(BaseModel):
    content: list[Word]


json_schema = json.dumps(Result.model_json_schema(), indent=2)
main_text = f"""
この写真は、日本の大学受験用英単語帳である「システム英単語」の1ページです。
この英単語帳をデータ化するために、このページを構造化して、指定されたスキーマのJSONとして出力してください。

この単語帳の仕様と、出力形式に関する注意は以下の通りです。
- この単語帳は、ページ上部に「MINIMAL PHRASES」という、重要フレーズ臭があります。今回はこれは扱わないため、完全に無視してOKです。
- 一般的な注意として、画像に存在する文字を、忠実に、一字一句同じように出力してください。
  - 画像が潰れていて/画質が悪くて識別不能な箇所は、文脈から判断してください。
- 単語ごとの構成はこうなっています:
  - 右側: 英単語、発音記号、単語にまつわる問題(sub_problem)
  - 左側: 日本語での意味、頻出する形、関連語、sub_problemの答えなど
- 右側に位置する、英単語をそのままwordに含めてください。
- 右側には、語に関する注意(notes)があることがあります。これらはnotesフィールドにリストとして含めてください。種類は以下の通りです。
  - 角丸線で囲んだ「多義」: 単語に複数の意味があることに注意
  - 角丸線で囲んだ「語法」: 語法や構文に注意
- 同じように、右側には、単語について注意する必要がある項目に関する問題が含まれることがあります。これらは、その右側に答えが位置しているため、その答えと併せてsub_problemsフィールドにリストとして含めてください。
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
    - このタイプの単語は、左側に発音記号があります。発音記号は普通に出力に含めた上で、「アクセントは何ですか」が問題文、発音記号を答えとするsub_problemにしてください。
  - 角丸線で囲んだ「語法」: 語法や構文に注意
    - これは、「語法や構文に注意」を問題文として、その語法や構文を答えとするsub_problemにしてください。
  - Q: それ以外の一般の問題
- 左側には、日本語の意味が含まれています。多義語の場合などは、①②などでいくつかの意味が列挙されている場合がありますが、これらはすべてanswerに含めてください。
- 日本語での意味の中でも、特に重要なのもは赤くハイライトされています。これらは<red>タグで囲むことで表現してください。
  - 単語本体だけでなく、赤くなっている部分全体を<red>で囲んでください。赤くない部分は決して囲まないでください。
- 日本語での意味における、「動」や「名」などの漢字を四角で囲んだ記号は、「動詞形」などという意味です。出力には、「動詞形」などの形で含めてください。
- 日本語での意味の説明において、存在する記号は以下の通りです。これらはそのままの記号で出力に含めてください(四角で囲んだ「動」などは単に漢字1文字を含めてください)。
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
- 単語の日本語表現(複数ある場合もある)と、それに直接含まれる説明のみをanswerに、それ以外の説明をdescriptionに含めてください。
- 右側、あるいは稀に左側に位置する発音記号(アクセントも含む)をphoneticに含めてください。

JSON schema:
{json_schema}
"""

external_client = AsyncOpenAI(
    api_key=os.getenv("OPEN_ROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

agent = Agent(
    name="Wordbook OCR",
    output_type=Result,
    model=OpenAIChatCompletionsModel(
        model="anthropic/claude-3.7-sonnet",
        openai_client=external_client,
    ),
    instructions="指示に従い、OCR/構造化を行ってください。なお、出力には結果のみを含め、それ以外のあらゆる情報や補足、コードブロックの記号などは一切含まないでください。"
)

result = Runner.run_sync(
    agent,
    [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": main_text,
                },
                {
                    "type": "input_image",
                    "detail": "high",
                    "image_url": process_image(Path("processed/2.png")),
                },
            ],
        },
    ],
)

for c in result.final_output.content:
    print(c.model_dump_json(indent=2))
    print("----------------")
