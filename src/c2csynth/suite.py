"""Run the replicated policy benchmark with one shared oracle."""
import argparse
from pathlib import Path
import subprocess
import sys


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=("smoke", "paper"), default="smoke")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args(argv)
    root = a.output.resolve()
    owner = root / "randomized/log04"

    def run(regime, number):
        output = root / regime / f"log{number:02d}"
        args = [sys.executable, "-m", "c2csynth.benchmark", "--preset", a.preset,
                "--regime", regime, "--log", str(number), "--output", str(output),
                "--device", a.device, "--threads", str(a.threads), "--resume"]
        if output != owner:
            args += ["--oracle-from", str(owner)]
        subprocess.run(args, check=True)

    run("randomized", 4)
    numbers = range(1, 11) if a.preset == "paper" else (1,)
    for regime in ("randomized", "observable", "latent"):
        for number in numbers:
            if regime != "randomized" or number != 4:
                run(regime, number)
    subprocess.run([sys.executable, "-m", "c2csynth.summarize", "--root", str(root)], check=True)


if __name__ == "__main__":
    main()
