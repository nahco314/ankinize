import argparse
import hashlib
import html
import json
import re
from pathlib import Path

import genanki
from google.cloud import texttospeech


MODEL_ID = 2025111401  # 固定（テンプレートが変わる場合のみ更新）
DEFAULT_DECK_ID_SEED = 942311  # デッキ名と合わせてハッシュ化して決定
TTS_CACHE_DIR = Path("tts")
VOICE_NAME = "en-US-Wavenet-F"


def sanitize_tag(s: str) -> str:
    """
    Anki タグは空白区切り。空白→_ にし、/ や \, 二重引用符など
    問題が出やすい文字を安全側に置換。
    """
    if not s:
        return "no_section"
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = s.replace("/", "_").replace("\\", "_").replace('"', "").replace("'", "")
    # タグは長すぎると扱いにくいので適度に丸める
    return s[:64]


def stars_html(level: int) -> str:
    """
    importance_level が 1 → * を1つ、2 → ** を2つ（赤）
    それ以外/未指定は空。
    """
    n = 0
    try:
        n = int(level)
    except Exception:
        n = 0
    n = max(0, min(n, 2))
    return f'<span class="importance">{"*" * n}</span>' if n > 0 else ""


def escape_preserving_red(text: str) -> str:
    """
    HTMLエスケープしつつ、<red>…</red> だけは赤字に変換。
    改行は <br>。
    """
    if text is None:
        return ""
    RED_OPEN = "###RED_OPEN###"
    RED_CLOSE = "###RED_CLOSE###"
    text = text.replace("<red>", RED_OPEN).replace("</red>", RED_CLOSE)
    text = html.escape(text)
    text = text.replace("\n", "<br>")
    text = text.replace(RED_OPEN, '<span class="red-text">')
    text = text.replace(RED_CLOSE, "</span>")
    return text


def render_phonetic(ph: str) -> str:
    if not ph:
        return ""
    return f'<span class="phonetic">{html.escape(ph)}</span>'


def render_meanings(meanings, f=False) -> str:
    """
    meanings: list[str] を <div> で縦に積む（<red>対応）。
    """
    if not meanings:
        return ""
    parts = [f"<div>{escape_preserving_red(m)}</div>" for m in meanings if m]

    if len(parts) == 1:
        return parts[0]
    else:
        res = parts[0]

        if f:
            res += "<br>"
        else:
            return res

        res += "".join(parts[1:])

        return res


def render_group_context(group: dict) -> str:
    """
    裏面の下部に載せる、同一 group の参照コンテンツ：
    - タイトル
    - リード（段落）
    - 単語一覧（左=単語+発音+重要度, 右=意味）
    """
    title = html.escape(group.get("title", ""))
    # lead は段落配列（改行→<br>）
    lead_parts = []
    for para in group.get("lead", []) or []:
        lead_parts.append(f"<p>{escape_preserving_red(para)}</p>")
    lead_html = "".join(lead_parts)

    # 単語一覧テーブル（2カラム）
    rows = []
    for w in group.get("words", []) or []:
        left = (
            f'<div class="gw-left">'
            f"{stars_html(w.get('importance_level'))}"
            f'<span class="gw-word">{html.escape(w.get("word", ""))}</span>'
            f"{render_phonetic(w.get('phonetic', ''))}"
            f"</div>"
        )
        right = f'<div class="gw-right">{render_meanings(w.get("meanings", []))}</div>'
        rows.append(f'<div class="gw-row">{left}{right}</div>')

    words_table = f'<div class="group-words">{"".join(rows)}</div>'

    return f"""
    <div class="group-context">
      <div class="group-title">{title}</div>
      {words_table}
      <div class="group-lead">{lead_html}</div>
    </div>
    """


def make_model() -> genanki.Model:
    """
    3フィールド構成：
      0: FrontWord（単語+発音+重要度）
      1: Meanings（意味ブロック）
      2: GroupContext（同一 group のタイトル/リード/全語一覧）
    """
    return genanki.Model(
        MODEL_ID,
        "Grouped Vocabulary v1",
        fields=[
            {"name": "FrontWord"},
            {"name": "Meanings"},
            {"name": "GroupContext"},
            {"name": "Audio"},
        ],
        templates=[
            {
                "name": "Card 1",
                "qfmt": """
<div class="card">
  <div class="front-question">{{FrontWord}}</div>
</div>
""",
                "afmt": """
<div class="card">
  <div class="front-question">{{FrontWord}}</div>
  <hr id="answer">
  <div class="meanings">{{Meanings}}</div>
  {{#Audio}}
  <div class="audio">{{Audio}}</div>
  {{/Audio}}
  <div class="ref-header">Group Reference</div>
  {{#GroupContext}}
  {{GroupContext}}
  {{/GroupContext}}
</div>
""",
            }
        ],
        css="""
/* ベース */
.card {
  font-family: "Helvetica Neue", Arial, "Hiragino Kaku Gothic ProN", "Yu Gothic", Meiryo, sans-serif;
  font-size: 16px;
  line-height: 1.6;
  color: #222;
  background-color: #fafafa;
  padding: 12px;
  border-radius: 10px;
}

/* 表面：単語行（中央寄せ・大きめ） */
.front-question {
  text-align: center;
  margin: 0.2em 0 0.4em 0;
  color: #24333e;
}

.front-question .gw {
  display: inline-flex;
  align-items: baseline;
  gap: 0.35em;
}

.front-question .importance {
  color: #e53935; /* 赤アスタリスク */
  font-weight: 700;
  letter-spacing: 0.05em;
}

.front-question .english {
  font-size: 1.85em;
  font-weight: 800;
}

.front-question .phonetic,
.phonetic {
  font-size: 0.95em;
  color: #6a7279;
}

/* 区切り線（上下の分け方） */
hr#answer {
  margin: 0.8em 0 0.9em 0;
  border: 0;
  border-top: 1px solid #d8dee4;
}

/* 意味（裏面の主コンテンツ） */
.meanings {
  font-size: 1.05em;
}

.red-text { color: #d32f2f; }

/* 参照セクション（スクロール領域） */
.ref-header {
  margin-top: 0.8em;
  font-size: 0.9em;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #7c8790;
}

.group-context {
  margin-top: 0.4em;
  max-height: 45vh;             /* 長い group でも崩れないように */
  overflow: auto;
  padding-top: 0.5em;
  border-top: 1px dashed #e0e3e7;
  background: #fff;
  border-radius: 6px;
}

/* group タイトルとリード */
.group-title {
  font-weight: 700;
  color: #37474f;
  margin: 0.1em 0 0.35em 0;
}

.group-lead p {
  margin: 0.35em 0;
  color: #4a4f55;
}

/* 単語一覧（2カラム） */
.group-words {
  display: grid;
  grid-template-columns: 1fr;
  gap: 8px;
  margin-top: 8px;
}

/* 各行は 2カラムに分割（モバイル→縦並び、広い画面→2列） */
.gw-row {
  display: grid;
  grid-template-columns: 1fr;
  gap: 10px;
  padding: 8px;
  border: 1px solid #eef1f4;
  border-radius: 8px;
  background: #fcfdff;
}

@media (min-width: 540px) {
  .gw-row {
    grid-template-columns: 0.42fr 0.58fr;
  }
}

/* 左：単語・発音・重要度 */
.gw-left {
  display: inline-flex;
  align-items: baseline;
  gap: 0.5em;
  min-width: 0;
}

.gw-left .importance {
  color: #e53935;
  font-weight: 700;
}

.gw-left .gw-word {
  font-weight: 700;
  color: #2b3a42;
  white-space: nowrap;
}

/* 右：意味（複数行OK） */
.gw-right {
  min-width: 0;
}

/* 表面の FrontWord を構成する内側（テンプレから入れやすく） */
.front-question .importance + .english { margin-left: 0.1em; }

.audio {
  margin-top: 0.6em;
}
""",
    )


def front_word_html(word: str, phonetic: str, level: int) -> str:
    return (
        '<div class="gw">'
        f"{stars_html(level)}"
        f'<span class="english">{html.escape(word)}</span>'
        f"{render_phonetic(phonetic)}"
        "</div>"
    )


def note_guid(section: str, group_title: str, word: str) -> str:
    base = f"{section}||{group_title}||{word}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]


def build_deck_id(deck_name: str) -> int:
    h = hashlib.sha1(f"{DEFAULT_DECK_ID_SEED}:{deck_name}".encode("utf-8")).hexdigest()
    # genanki Deck ID は 32bit 程度でOK
    return int(h[:8], 16)


def build_notes(data, deck, model, media_files: set[Path]):
    """
    data: JSON のトップ（list[section]）を想定
    """
    if not isinstance(data, list):
        raise ValueError("トップレベルは配列（sections の配列）を想定しています。")

    for sec in data:
        section_heading = sec.get("section_heading", "")
        section_tag = sanitize_tag(section_heading)

        for group in sec.get("groups", []) or []:
            group_title = group.get("title", "")
            group_context = render_group_context(group)

            for w in group.get("words", []) or []:
                word = w.get("word", "")
                if not word:
                    continue
                meanings_html = render_meanings(w.get("meanings", []), True)
                front_html = front_word_html(
                    word=word,
                    phonetic=w.get("phonetic", ""),
                    level=w.get("importance_level", 0),
                )

                audio_field, media_path = build_audio_field(word)
                if media_path:
                    media_files.add(media_path)

                note = genanki.Note(
                    model=model,
                    fields=[
                        front_html,
                        meanings_html,
                        group_context,
                        audio_field,
                    ],
                    guid=note_guid(section_heading, group_title, word),
                    tags=[section_tag],  # section をタグへ
                )
                deck.add_note(note)


def normalized_tts_text(text: str) -> str:
    if not text:
        return ""
    return text.replace("<red>", "").replace("</red>", "").strip()


_tts_client: texttospeech.TextToSpeechClient | None = None


def get_tts_client() -> texttospeech.TextToSpeechClient:
    global _tts_client
    if _tts_client is None:
        _tts_client = texttospeech.TextToSpeechClient()
    return _tts_client


def generate_tts_file(text: str, lang_code: str = "en-US") -> Path:
    clean_text = normalized_tts_text(text)
    if not clean_text:
        raise ValueError("TTS text is empty")

    hash_val = hashlib.md5(clean_text.encode("utf-8")).hexdigest()[:10]
    TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_file = TTS_CACHE_DIR / f"{hash_val}.mp3"
    if out_file.exists():
        return out_file

    client = get_tts_client()
    synthesis_input = texttospeech.SynthesisInput(text=clean_text)
    voice = texttospeech.VoiceSelectionParams(
        language_code=lang_code,
        name=VOICE_NAME,
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL,
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config,
    )
    out_file.write_bytes(response.audio_content)
    return out_file


def build_audio_field(text: str) -> tuple[str, Path | None]:
    try:
        tts_path = generate_tts_file(text)
    except Exception as exc:
        print(f"[WARN] Failed to generate TTS for '{text}': {exc}")
        return "", None
    return f"[sound:{tts_path.name}]", tts_path


def main():
    input_path = Path("./res.json")
    if not input_path.exists():
        raise FileNotFoundError(f"JSON が見つかりません: {input_path}")

    data = json.loads(input_path.read_text(encoding="utf-8"))

    deck_id = build_deck_id("鉄壁")
    deck = genanki.Deck(deck_id, "鉄壁")
    model = make_model()

    media_files: set[Path] = set()
    build_notes(data, deck, model, media_files)

    pkg = genanki.Package(deck)
    if media_files:
        pkg.media_files = sorted(str(path) for path in media_files)
    pkg.write_to_file("output.apkg")
    print(f"Done")


if __name__ == "__main__":
    main()
