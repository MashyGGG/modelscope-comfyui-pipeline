import pytest

from pipeline.common import ROOT, config, paths
from pipeline import install


def test_cuda_wheels_and_ordinary_dependencies_use_separate_indexes(tmp_path, monkeypatch):
    c = config(overrides=[f"workspace_root={tmp_path.as_posix()}"])
    calls = []
    monkeypatch.setattr(install, "run", lambda command: calls.append([str(x) for x in command]))
    install.install_packages(c)
    pip_upgrade, wheels, dependencies, check = calls
    assert pip_upgrade[pip_upgrade.index("--index-url")+1] == c["pypi_index_url"]
    assert wheels[wheels.index("--index-url")+1] == "https://download.pytorch.org/whl/cu126"
    assert "--no-deps" in wheels and str(ROOT / "constraints.txt") in wheels
    assert {part for part in wheels if "==" in part} == {
        "torch==2.8.0+cu126", "torchvision==0.23.0+cu126", "torchaudio==2.8.0+cu126"}
    assert dependencies[dependencies.index("--index-url")+1] == c["pypi_index_url"]
    assert "--no-deps" not in dependencies and str(ROOT / "constraints.txt") in dependencies
    assert str(ROOT / "requirements.txt") in dependencies
    assert str(paths(c)["comfy"] / "requirements.txt") in dependencies
    assert check[-2:] == ["pip", "check"]


def test_dependency_install_failure_stops_before_success_check(monkeypatch):
    calls = []
    def command(args):
        calls.append(args)
        if "-r" in args:
            raise RuntimeError("dependency connection failed")
    monkeypatch.setattr(install, "run", command)
    with pytest.raises(RuntimeError, match="connection failed"):
        install.install_packages(config())
    assert len(calls) == 3 and not any(args[-1] == "check" for args in calls)


def test_package_index_and_retries_can_be_configured(monkeypatch):
    c = config(overrides=["pypi_index_url=https://pypi.org/simple", "pip_timeout=15", "pip_retries=1"])
    calls = []
    monkeypatch.setattr(install, "run", lambda args: calls.append(args))
    install.install_packages(c)
    assert calls[0][calls[0].index("--index-url")+1] == "https://pypi.org/simple"
    for args in calls[:3]:
        assert args[args.index("--timeout")+1] == "15"
        assert args[args.index("--retries")+1] == "1"


@pytest.mark.parametrize("value", ["pypi_index_url=http://example.com/simple", "pip_timeout=0", "pip_retries=-1"])
def test_invalid_package_network_options_rejected(value):
    with pytest.raises(ValueError):
        config(overrides=[value])
