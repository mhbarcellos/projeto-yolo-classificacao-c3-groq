from __future__ import annotations

import subprocess
import sys


def main() -> None:
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--onefile",
        "--windowed",
        "--name",
        "ReconhecimentoYOLO",
        "run_app.py",
    ]
    print("Executando:", " ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
