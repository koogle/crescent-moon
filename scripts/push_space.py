"""Assemble the demo Space and push it to Hugging Face.

    uv run python scripts/push_space.py --examples artifacts/demo-examples --push

The Space is `space/` (app, camera page, metadata), the package `src/crescent_moon`,
the browser runtime `web/*.js` and example images (`--examples`, a folder of JPEGs). The
assembled folder is `--out`; `--push` uploads it to `--repo`, creating the Space on
ZeroGPU when it does not exist. Without `--push` the folder can be run locally:
`python artifacts/space/app.py`.
"""

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNTIME = ("kernels.js", "model.js", "select.js")


def assemble(out, examples):
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(ROOT / "space", out)
    shutil.copytree(
        ROOT / "src" / "crescent_moon",
        out / "crescent_moon",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (out / "web").mkdir()
    for name in RUNTIME:
        shutil.copy(ROOT / "web" / name, out / "web" / name)
    (out / "examples").mkdir()
    for image in sorted(examples.glob("*.jpg")):
        shutil.copy(image, out / "examples" / image.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "space")
    parser.add_argument("--repo", default="jfrick/crescent-moon")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    assemble(args.out, args.examples)
    if args.push:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(
            args.repo,
            repo_type="space",
            space_sdk="gradio",
            space_hardware="zero-a10g",
            exist_ok=True,
        )
        api.upload_folder(
            repo_id=args.repo,
            repo_type="space",
            folder_path=args.out,
            delete_patterns=["*"],
            commit_message="Update the demo from the repository",
        )
        print(f"https://huggingface.co/spaces/{args.repo}")


if __name__ == "__main__":
    main()
