import asyncio
import html
from pathlib import Path

import genanki

from model import FinalResult, Word, MetaData
from utils import retrynize

my_model = genanki.Model(
    1607392320,  # 新しいID
    "Simple Model v2",
    fields=[
        {"name": "english"},
        {"name": "japanese"},
        {"name": "casual_tag"},
        {"name": "examples"},
        {"name": "additional"},
        {"name": "meta_png"},
    ],
    templates=[
        {
            "name": "Card 1",
            "qfmt": """
    <div class="card">
      <!-- 英単語(メイン) -->
      <div class="front-question">{{english}}</div>

      <!-- casual_tag が空でなければ表示 -->
      {{#casual_tag}}
      <div class="notes">{{casual_tag}}</div>
      {{/casual_tag}}
    </div>
    """,
            "afmt": """
    <div class="card">
      <!-- 裏面でも問題文（表面）を再掲 -->
      <!-- 英単語(メイン) -->
      <div class="front-question">{{english}}</div>

      <!-- casual_tag が空でなければ表示 -->
      {{#casual_tag}}
      <div class="notes">{{casual_tag}}</div>
      {{/casual_tag}}
      <hr id="answer">

      <div class="answer-section">
        {{#japanese}}<div class="japanese">{{japanese}}</div>{{/japanese}}

        {{#examples}}
        <div class="description">{{examples}}</div>
        {{/examples}}

        {{#additional}}
        <div class="additional">{{additional}}</div>
        {{/additional}}
      </div>
    </div>
    """,
        },
    ],
    css="""
    /* カード全体 */
    .card {
      font-family: "Helvetica Neue", Arial, sans-serif;
      font-size: 16px;
      line-height: 1.6;
      color: #333;
      background-color: #fafafa;
      padding: 12px;
      border-radius: 8px;
    }

    /* 英単語(メイン) */
    .front-question {
      text-align: center;  /* 中央揃え */
      font-size: 1.8em;    /* 大きめにして主役感を出す */
      font-weight: bold;
      margin-bottom: 0.5em;
      color: #2c3e50;
    }

    /* notes */
    .notes {
      font-size: 0.9em;
      color: #555;
      margin-top: 0.5em;
      padding: 0.5em;
      border: 1px dashed #ccc;
      background-color: #fefefe;
    }

    /* 裏面の全体 */
    .answer-section {
      /* 必要に応じてスタイルを追加 */
    }

    /* 日本語訳など(赤色はやめて太字のみ) */
    .japanese {
      font-size: 1.2em;
      font-weight: bold;
      margin-top: 0.5em;
      color: #333; /* 赤色を廃止 */
    }

    .red-text {
      color: red;
    }

    /* 例文 */
    .description {
      margin-top: 1em;
      color: #444;
    }

    /* 追加情報 */
    .additional {
      margin-top: 1em;
      padding: 0.75em;
      border-left: 4px solid #3498db;
      background: #ecf9ff;
      color: #444;
    }

    /* 区切り線 */
    hr#answer {
      margin: 1em 0;
      border: 0;
      border-top: 1px solid #ccc;
    }
    """,
)


def escape(string: str) -> str:
    """
    HTMLエスケープ & 改行→<br> 変換しつつ、
    <red>〜</red> で囲まれた部分だけは赤字にする。
    """
    if not string:
        return ""

    # 1) プレースホルダで <red> と </red> を一旦退避
    RED_OPEN = "###RED_OPEN###"
    RED_CLOSE = "###RED_CLOSE###"

    # <red>... </red> をプレースホルダに置換
    string = string.replace("<red>", RED_OPEN).replace("</red>", RED_CLOSE)

    # 2) HTMLエスケープ
    string = html.escape(string)

    # 3) 改行文字を <br> に
    string = string.replace("\n", "<br>")

    # 4) プレースホルダを <span style="color:red;"> に置換
    string = string.replace(RED_OPEN, '<span class="red-text">')
    string = string.replace(RED_CLOSE, "</span>")

    return string


async def gen_note_async(word: Word, metadata: MetaData) -> genanki.Note:
    """
    Word の情報からノートを作る（非同期構造は保持）
    """
    # casualタグ
    casual_tag = ""
    if word.casual:
        casual_tag = "口語"

    # 例文を組み立て
    examples = ""
    if word.example_en or word.example_ja:
        if word.example_en:
            examples += f"<div>{escape(word.example_en)}</div>"
        if word.example_ja:
            examples += f"<div>{escape(word.example_ja)}</div>"

    # Noteオブジェクト生成
    note = genanki.Note(
        model=my_model,
        fields=[
            escape(word.word),  # english
            escape(word.answer),  # japanese
            casual_tag,  # casual_tag
            examples,  # examples
            escape(word.additional).replace("\n", "<br>") if word.additional else "",  # additional
            str(metadata.input_image_name) if hasattr(metadata, 'input_image_name') else "",  # meta_png
        ],
    )
    return note


async def main():
    base = Path("./outputs-2-normal")
    my_deck = genanki.Deck(2059400111, "速読英熟語")

    # JSON を走査して Word, MetaData を取得する処理
    all_tasks = []
    for i in range(144):
        normal_path = base / f"{i}.json"
        minimal_path = base / f"{i}-minimal.json"

        for path in [normal_path, minimal_path]:
            if path.exists():
                content = path.read_text(encoding="utf-8")
                final_res = FinalResult.model_validate_json(content)

                # 複数 Word を async ノート生成
                for c in final_res.result.content:
                    task = asyncio.create_task(gen_note_async(c, final_res.metadata))
                    all_tasks.append(task)

    # ここで一気にノート生成
    notes = await asyncio.gather(*all_tasks)

    # できあがった Note をデッキに追加
    for note in notes:
        my_deck.add_note(note)

    # パッケージ作成（メディアファイルなし）
    package = genanki.Package(my_deck)

    # apkgファイル出力
    package.write_to_file("output.apkg")
    print("Done: output.apkg を生成しました。")


############################
# スクリプト実行部
############################

if __name__ == "__main__":
    asyncio.run(main())