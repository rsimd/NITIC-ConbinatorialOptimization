# エージェント向けメモ（本リポジトリの資料作成）

人間および AI エージェントが、学生向け Jupyter ノートや MyST/Jupyter Book サイトを増補するときに揃えたい運用情報である。

## 公開 URL と内部リンク（重要）

この教材は GitHub Actions により **GitHub Pages** に公開される。**ノート間のジャンプだけ** Markdown の相対リンク（例: `[第3節](#sec-...)`、`lecture04_pso.ipynb`）にすると、**ビルド後の静的 HTML では期待どおりに飛ばない**ことがある。

学生配布ファイルの本文では、**サイトルートからのフルパス URL** で内部リンクを書くことを推奨する。

- **サイトベース URL（現在の既定）**: `https://rsimd.github.io/NITIC-ConbinatorialOptimization/`
- **公開パス（MyST の book-theme 既定）**: ソースの `_` が `-` に置き換わった **スラグ** で、ページは **`lecture05-aco-tsp/index.html`** のように **ディレクトリ＋ `index.html`** として出力される。**ブラウザ上の URL は** `…/lecture05-aco-tsp` または末尾 `/` が付いた形式である（`**lecture05_aco_tsp.html` のようなフラットファイル名ではない**）。
- **ページ内アンカー**: MyST ではセル冒頭などに `(sec:my-anchor)=` と書くと、HTML 側は多くの場合 **`#sec-my-anchor`** となる（ハイフン区切り）。

例（PSO の第 3 節へリンクする場合）。

```markdown
https://rsimd.github.io/NITIC-ConbinatorialOptimization/lecture04-pso#sec-pso-ch3
```

ローカルで **GitHub Pages と同じパス** で `file://` やローカルサーバーを試すときは、[README.md](README.md) のとおり **`BASE_URL=/NITIC-ConbinatorialOptimization`** を付けて HTML ビルドする。

## 先頭セルと Google Colab

学生に `.ipynb` を直接渡す運用であるため、**第 1 セル（タイトル直後）** に Colab バッジを置く。

- バッジ画像: `https://colab.research.google.com/assets/colab-badge.svg`
- リンク先（`master` が既定ブランチの場合の例）。

```markdown
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rsimd/NITIC-ConbinatorialOptimization/blob/master/<ノートファイル名>.ipynb)
```

組織・ブランチ名・ファイル名が変わったら **バッジ URL も追随** すること。

## 目次への登録

新しいノートを公開サイトのサイドバーに出すには、[myst.yml](myst.yml) の `project.toc` に `file:` 行で追加する。あわせて [index.md](index.md) の箇条書きに **第〇回** としてリンクを追加する。公開 URL は **`https://<サイト>/lecture05-aco-tsp`** のような **スラグ** である（詳細はこの文書の「公開 URL と内部リンク」を参照）。

ノート `lecture05_aco_tsp.ipynb` は **複数コードセル＋説明マークダウンに分割済み** である。そのまま手で編集する運用でもよい。`scripts/gen_lecture05_notebook.py` を無差別に実行すると **単一大コードセルの形に置き換わり**、この分割や説明文が失われる恐れがある点にだけ注意すること。

## 文体・トーン

ディレクトリの方針に合わせ、**説明は細かめ**、語り口は優しめでもよく、文末は **`だ／である` 調** を基本とする。ときどき **読者への問い**（「皆さんはどう思うだろうか？」など）を入れてよい。

## コード方針（Python 講義ノート）

- **説明や補足は、対応するコードセルの直前にある Markdown セルにのみ書く**こと。見出し用の親セクション（§4 の直後など）に実装詳説を繰り返さないほうが、学生が読み流しやすい。
- **単一問題に絞ったシンプルなクラス実装**（アルゴリズム本体と可視化を分離しすぎない程度に整理）が望ましい。
- **matplotlib** と **numpy** は [pyproject.toml](pyproject.toml) に依存としてある。Colab 単体実行を想定するセルには、必要なら `pip install ...` を短く書く（本リポジトリでは **`uv`** を使うが、ノート本文は Google Colab 利用者にも読めるように両立させる）。
- **数式・疑似コード** を本文に明示する。

## MCP 関連（ユーザー設定）

ユーザー設定により「Gemini CLI / Claude Code MCP を活用する」「日本語出力はだ・である調」などがある。本作業では **Claude Code / Gemini MCP サーバーへは接続していない**。利用可能な環境では、ユーザー指示に従いこれらを併用してよい。
