import asyncio
import concurrent.futures
import html
from collections import Counter
from pathlib import Path
import hashlib
import os
import re

import genanki
from google.cloud import texttospeech

from model import FinalResult, Word, MetaData
from utils import retrynize


TTS_SEMAPHORE = asyncio.Semaphore(5)


@retrynize
def generate_tts_file_gcloud_sync(text: str, lang_code: str = "en-US") -> Path:
    """
    Google Cloud Text-to-Speech を使って .mp3 を生成し、ファイルパスを返す同期関数。
    既に同一テキストの音声ファイルがあれば再生成しない (キャッシュ)。
    """

    text = text.replace("<red>", "").replace("</red>", "")  # 赤字タグは削除

    # テキストに対してハッシュを作り、ファイル名を一意に。
    hash_val = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
    out_file = Path(f"tts/{hash_val}.mp3")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if out_file.exists():
        # 既に作ったファイルがあれば再利用
        return out_file

    # =============================
    # 1) TTS クライアント初期化
    # =============================
    # 環境変数 GOOGLE_APPLICATION_CREDENTIALS が設定されている場合は自動検出。
    # あるいは下記のようにファイルを直接指定してもOK:
    # client = texttospeech.TextToSpeechClient.from_service_account_file("path/to/key.json")
    client = texttospeech.TextToSpeechClient()

    # =============================
    # 2) 音声合成リクエスト作成
    # =============================
    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code=lang_code,
        name="en-US-Wavenet-F",
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL,
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        # speaking_rate=1.0, pitch=0.0 など調整可能
    )

    # =============================
    # 3) 音声合成リクエスト実行
    # =============================
    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config,
    )

    # =============================
    # 4) MP3ファイルに保存
    # =============================
    with open(out_file, "wb") as f:
        f.write(response.audio_content)

    return out_file

async def generate_tts_file_gcloud_async(text: str, lang_code: str = "en-US") -> Path:
    """
    generate_tts_file_gcloud_sync をスレッドプールで非同期実行し、並列で TTS 処理する。
    """
    loop = asyncio.get_running_loop()
    async with TTS_SEMAPHORE:
        # セマフォで同時実行数を制限
        return await loop.run_in_executor(None, generate_tts_file_gcloud_sync, text, lang_code)


my_model = genanki.Model(
    1607392319,
    "Simple Model",
    fields=[
        {"name": "english"},
        {"name": "japanese"},
        {"name": "description"},
        {"name": "sub_problem_statements"},
        {"name": "sub_problem_answers"},
        {"name": "notes"},
        {"name": "phonetic"},
        {"name": "meta_png"},
        {"name": "audio"},
    ],
    templates=[
        {
            "name": "Card 1",
            "qfmt": """
    <div class="card">
      <!-- 英単語(メイン) -->
      <div class="front-question">{{english}}</div>

      <!-- sub_problem_statements が空でなければ表示 -->
      {{#sub_problem_statements}}
      <div class="sub-problem-statements">{{sub_problem_statements}}</div>
      {{/sub_problem_statements}}

      <!-- notes が空でなければ表示 -->
      {{#notes}}
      <div class="notes">{{notes}}</div>
      {{/notes}}
    </div>
    """,
            "afmt": """
    <div class="card">
      <!-- 裏面でも問題文（表面）を再掲 -->
      {{FrontSide}}
      <hr id="answer">

      <div class="answer-section">
        {{#japanese}}<div class="japanese">{{japanese}}</div>{{/japanese}}
        {{#description}}<div class="description">{{description}}</div>{{/description}}

        {{#sub_problem_answers}}
        <div class="sub-problem-answers">{{sub_problem_answers}}</div>
        {{/sub_problem_answers}}

        {{#phonetic}}
        <div class="phonetic">{{phonetic}}</div>
        {{/phonetic}}
        
        {{#audio}}
        <div class="audio">{{audio}}</div>
        {{/audio}}
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

    /* サブ問題など */
    .sub-problem-statements,
    .sub-problem-answers {
      margin: 1em 0;
      padding: 0.75em;
      border-left: 4px solid #3498db;
      background: #ecf9ff;
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

    /* 説明 */
    .description {
      margin-top: 1em;
      color: #444;
    }

    /* フォネティック */
    .phonetic {
      margin-top: 1em;
      font-style: italic;
      color: #7f8c8d;
    }

    /* 区切り線 */
    hr#answer {
      margin: 1em 0;
      border: 0;
      border-top: 1px solid #ccc;
    }
    """,
)


class IdManager:
    def __init__(self):
        self.counter = Counter()

    def get_id(self, name) -> int:
        self.counter[name] += 1
        return hash(f"{self.counter[name]}-{name}")


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
    # （正規表現を使わず単純置換でもOKですが、一度に全部→戻す形が手軽）
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
    Word の情報からノートを作る。
    ただし TTS ファイル生成 (gTTS) を非同期化。
    """
    # サブ問題
    if not word.sub_problems:
        sub_problem_statements = ""
        sub_problem_answers = ""
    elif len(word.sub_problems) == 1:
        sub_problem_statements = escape(word.sub_problems[0].question)
        sub_problem_answers = escape(word.sub_problems[0].answer)
    else:
        sub_problem_statements = ""
        sub_problem_answers = ""
        id_manager = IdManager()
        for sp in word.sub_problems:
            sub_problem_statements += f"<div>{escape(sp.question)}</div>"
            sub_problem_answers += f"<div>{escape(sp.answer)}</div>"

    # notes
    notes = ""
    for n in word.notes:
        notes += f"<div>{escape(n.value)}</div>"

    # TTS の音声ファイルを生成（非同期）
    tts_file_path = await generate_tts_file_gcloud_async(word.word)

    # Anki フィールドに [sound:xxx.mp3] を仕込む
    audio_field = f"[sound:{tts_file_path.name}]"  # パスのうちファイル名のみ指定

    # Noteオブジェクト生成
    note = genanki.Note(
        model=my_model,
        fields=[
            escape(word.word),  # english
            escape(word.answer),  # japanese
            escape(word.description).replace("\n", "<br>"),  # description
            sub_problem_statements,  # sub_problem_statements
            sub_problem_answers,  # sub_problem_answers
            notes,  # notes
            escape(word.phonetic),  # phonetic
            str(metadata.input_image_name),  # meta_png
            audio_field  # audio
        ],
    )
    return note


############################
# メインの処理を async 化
############################

async def main():
    base = Path("./outputs-0-normal")
    my_deck = genanki.Deck(2059400110, "システム英単語 Basic (test)")

    # JSON を走査して Word, MetaData を取得する処理（例）
    all_tasks = []
    for i in range(272):
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

    # ここで一気に TTS を並列実行させ、Noteを作る
    notes = await asyncio.gather(*all_tasks)

    # できあがった Note をデッキに追加
    for note in notes:
        my_deck.add_note(note)

    # メディアファイル（生成された mp3）を全て追加
    # TTSをキャッシュしているので、"tts"フォルダ内すべてをメディアとして含める方法が楽です
    tts_dir = Path("tts")
    media_files = [str(mp3) for mp3 in tts_dir.glob("*.mp3")]

    package = genanki.Package(my_deck)
    package.media_files = media_files

    # apkgファイル出力
    package.write_to_file("output.apkg")
    print("Done: output.apkg を生成しました。")


############################
# スクリプト実行部
############################

if __name__ == "__main__":
    asyncio.run(main())
