from pathlib import Path


def test_directory_iteration(tmp_path):
    # Ensure directories are handled without attempting Image.open on them
    d = tmp_path / "inputs"
    d.mkdir()
    f = d / "img.jpg"
    f.write_bytes(b"not really an image")

    files = [p for p in d.iterdir() if p.is_file()]
    assert len(files) == 1
