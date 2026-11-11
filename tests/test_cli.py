from churnml.cli import main


def test_generate_then_train(tmp_path, capsys):
    data = tmp_path / "data"
    assert main(["generate", "--out", str(data), "--customers", "600"]) == 0
    uri = f"sqlite:///{tmp_path / 'ml.db'}"
    code = main(["train", "--data", str(data), "--tracking-uri", uri])
    assert code == 0
    assert "best model:" in capsys.readouterr().out


def test_missing_data_folder_exits_two(tmp_path, capsys):
    code = main(["train", "--data", str(tmp_path / "nope")])
    assert code == 2
    assert "churnml: error" in capsys.readouterr().err
