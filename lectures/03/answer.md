# 第3回 解答

## 演習

- 第1回（素のLLM）とエージェントの回答の違い：
エージェントでは、最適なツール呼び出しプログラムを実行し、質問の回答に必要な情報を繰り返し探し、回答に反映させるのが、素のLLMとの最大の違いである。

- `--trace` で見えたツール呼び出し（ツール名と引数）：
[tool call]   search_glossary({'query': 'LoRA'})
[tool call]   search_glossary({'query': 'QLoRA'})
[tool return] search_glossary -> [{"term": "LoRA", "lecture": 2, "definition": "事前学習済みの重みは固定したまま、重みの更新分を2つの小さな行列の積（低ランク行列）で表し、その小さな行列だけを学習するPEFT手法。学習するパラ…
[tool return] search_glossary -> [{"term": "QLoRA", "lecture": 2, "definition": "ベースモデルの重みを4ビットなどに量子化してメモリを節約したうえで、LoRAで学習する手法。より大きなモデルを小さなGPUで扱える。本講座の第2…
[usage] RunUsage(cost=Decimal('0.0008644'), details={'text_prompt_tokens': 1198}, input_text_tokens=1198, input_tokens=1198, output_tokens=202, requests=2, tool_calls=2)

- 1つの質問での LLM 呼び出し回数（`requests`）：
2

- 追加したツールと、docstring に書いた説明：
追加したツール：make_quiz

docstring に書いた説明：指定した回の用語から1つを選び、確認問題と答えを返す。「第2回の確認問題を出して」のように、復習用の問題を求められたときに使う。

## 宿題

- 取り組んだレベル：実装
- やったこと：
学習記録をUTF-8のテキストとして .md ファイルに保存するツールを作成し、Pydantic AIのHuman-in-the-Loopによる保存前の承認機能を実装した。
- 結果：
承認時のみ学習記録が保存され、拒否時には保存されないことを確認した。pytestによるテストもすべて成功した。
- 気づき・疑問：
requires_approval=True によってツールの実行を一時停止でき、承認結果と会話履歴を渡すことで処理を再開できることを理解した。