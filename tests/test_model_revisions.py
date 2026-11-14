import hashlib

import pytest
from modelscope.hub.api import HubApi
from modelscope.hub.errors import NotExistError
from modelscope.hub import file_download

from pipeline.common import config
from pipeline import models


@pytest.mark.parametrize("repo", ["AI-ModelScope/stable-diffusion-xl-base-1.0", "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"])
def test_pinned_sdk_rejects_commit_even_when_present_as_branch_commit(repo, monkeypatch):
    # Real SDK 1.29.2 resolver; only its HTTP branch listing is replaced with fixture data.
    api = HubApi()
    commit = "a"*40
    branches = [{"Revision": "master", "CommitId": commit, "CreatedAt": 1}]
    monkeypatch.setattr(api, "get_model_branches_and_tags_details", lambda *a, **kw: (branches, []))
    with pytest.raises(NotExistError, match="has no revision"):
        api.get_valid_revision(repo, revision=commit, endpoint="https://modelscope.cn")
    assert api.get_valid_revision(repo, revision="master", endpoint="https://modelscope.cn") == "master"


def fixture(tmp_path):
    c = config(overrides=[f"workspace_root={tmp_path.as_posix()}", "download_attempts=1"])
    path = tmp_path / "sdk-cache.bin"
    path.write_bytes(b"exact-pinned-fixture")
    item = {"key": "test", "repo": "test/repo", "revision": "a"*40, "sdk_revision": "master",
            "file": "model.bin", "directory": "vae", "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return c, item, path


def test_sdk_uses_explicit_branch_after_both_metadata_checks(tmp_path, monkeypatch):
    c, item, path = fixture(tmp_path)
    monkeypatch.delenv("MODELSCOPE_API_TOKEN", raising=False)
    events = []
    monkeypatch.setattr(models, "remote_metadata", lambda m: events.append(m["revision"]))
    def sdk(repo, **kwargs):
        assert repo == item["repo"] and kwargs["revision"] == "master"
        events.append("SDK download")
        return str(path)
    monkeypatch.setattr(file_download, "model_file_download", sdk)
    models.download(c, [item])
    assert events == [item["revision"], "master", "SDK download"]
    assert models.valid(models.target(c, item), item)


def test_branch_metadata_drift_prevents_download(tmp_path, monkeypatch):
    c, item, path = fixture(tmp_path)
    def metadata(m):
        if m["revision"] == "master":
            raise RuntimeError("SDK branch metadata no longer matches pinned hash")
    monkeypatch.setattr(models, "remote_metadata", metadata)
    monkeypatch.setattr(file_download, "model_file_download", lambda *a, **kw: pytest.fail("must not download changed weights"))
    with pytest.raises(RuntimeError, match="Model download failed"):
        models.download(c, [item])
    assert not models.target(c, item).exists()


def test_downloaded_branch_file_must_still_match_pinned_hash(tmp_path, monkeypatch):
    c, item, path = fixture(tmp_path)
    path.write_bytes(b"x" * item["size"])
    monkeypatch.delenv("MODELSCOPE_API_TOKEN", raising=False)
    monkeypatch.setattr(models, "remote_metadata", lambda _: None)
    monkeypatch.setattr(file_download, "model_file_download", lambda *a, **kw: str(path))
    with pytest.raises(RuntimeError, match="Model download failed"):
        models.download(c, [item])
    assert not models.target(c, item).exists()
