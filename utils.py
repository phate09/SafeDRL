from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent


def get_save_dir() -> str:
    return str(get_project_root().joinpath("save"))
