"""素のLLMへの質問処理のテスト（LLMもAPIキーも使わない）。"""

from app import ask as ask_module


class FakeChat:
    def __init__(self):
        self.questions = []

    def send_message(self, question):
        self.questions.append(question)
        return type("Response", (), {"text": f"回答: {question}", "usage_metadata": None})()


class FakeClient:
    def __init__(self, chat):
        self.chat = chat
        self.chats = self

    def create(self, *, model):
        assert model == ask_module.MODEL
        return self.chat


def test_main_keeps_chat_for_follow_up_questions(monkeypatch, capsys):
    chat = FakeChat()

    def create_client(*, api_key):
        assert api_key == "test-key"
        return FakeClient(chat)

    monkeypatch.setattr(ask_module.genai, "Client", create_client)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr("sys.argv", ["app.ask", "最初の質問"])
    questions = iter(["追加の質問", ""])

    def answer_prompt(prompt):
        assert "終了するには空行" in prompt
        return next(questions)

    monkeypatch.setattr("builtins.input", answer_prompt)

    ask_module.main()

    assert chat.questions == ["最初の質問", "追加の質問"]
    output = capsys.readouterr().out
    assert "回答: 最初の質問" in output
    assert "回答: 追加の質問" in output
