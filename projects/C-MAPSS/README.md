# C-MAPSS: VS Code Notebook workflow

VS CodeのJupyter Notebook上で、C-MAPSSのEDAとRUL予測をセル単位で対話的に試すためのプロジェクトです。

## 推奨する学習順序

1. `notebooks/01_eda.ipynb`: データの形、品質、寿命、センサーの特徴を確認
2. `notebooks/02_baseline.ipynb`: 特徴量作成、モデル比較、公式test評価

ルートの`C-MAPSS.ipynb`は、EDAから学習までを一冊にまとめた実行済み参考版です。新しく実験するときは`notebooks/`の2冊を推奨します。

## 初回セットアップ

1. VS Codeで`C-MAPSS`フォルダ自体を開きます。
2. 推奨拡張機能のPython、Pylance、Jupyterをインストールします。
3. `Ctrl+Shift+P` → `Python: Select Interpreter`で、PCにインストール済みのPython 3.11以上を選びます。
4. `Ctrl+Shift+P` → `Tasks: Run Task` → `C-MAPSS: 1. 仮想環境を作成`を実行します。
5. `C-MAPSS: 2. Notebook用ライブラリをインストール`を実行します。
6. `notebooks/01_eda.ipynb`を開きます。
7. Notebook右上の「カーネルの選択」→「Python環境」→`.venv\Scripts\python.exe`を選びます。

## Notebookの基本操作

- `Shift+Enter`: 現在のセルを実行して次へ進む
- `Ctrl+Enter`: 現在のセルだけを実行
- `Esc`の後に`A`: 上へセルを追加
- `Esc`の後に`B`: 下へセルを追加
- 上部の「すべて実行」: カーネルを選んだ後、全セルを順番に実行
- 上部の「すべてクリア」: 出力だけを消して、コードは残す

Notebookの作業フォルダはVS Code設定でリポジトリ直下に固定しています。そのため、データや`src/cmapss`を安定して読み込めます。

## 対話的な実験

`02_baseline.ipynb`の最初の設定セルには次の変数があります。

    DATASET = "FD001"
    RUL_CAP = 125
    ROLLING_WINDOW = 5
    N_TREES = 250

値を変え、変更したセル以降を再実行するとモデルの違いを比較できます。結果保存時は`EXPERIMENT_NAME`も変更すると、過去の実験を上書きしません。

## Pythonファイルとの役割分担

- Notebook: グラフを見ながら仮説を試す場所
- `src/cmapss/data.py`: データ読み込みとRUL計算
- `src/cmapss/features.py`: 再利用する特徴量処理
- `src/cmapss/metrics.py`: 評価指標
- `main.py`: Notebookを開かず一括実行したい場合の入口

共通処理を修正したときは、Notebookの「カーネルを再起動」後にセルを再実行すると変更が反映されます。

## カーネルが表示されない場合

VS Codeのターミナルで次を実行し、その後Notebookを開き直します。

    .\.venv\Scripts\python.exe -m pip install -r requirements.txt

それでも表示されない場合は、`Ctrl+Shift+P` → `Developer: Reload Window`を実行してください。
