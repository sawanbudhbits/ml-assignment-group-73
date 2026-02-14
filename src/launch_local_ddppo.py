from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import tempfile
from typing import List


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local multi-process launcher for src.train_ddppo using file:// init_method (avoids macOS socket/IPv6 issues)."
    )
    parser.add_argument("--nproc", type=int, default=2, help="Number of worker processes to launch")
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to use (defaults to current interpreter)",
    )

    # Everything after '--' is forwarded to train_ddppo.
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    nproc = int(args.nproc)
    if nproc < 1:
        raise ValueError("--nproc must be >= 1")

    # Create a shared file store for rendezvous (avoids TCPStore + hostname/IPv6 issues).
    # IMPORTANT: init_method must be of the form file:///abs/path.
    fd, store_path = tempfile.mkstemp(prefix="ddppo_store_", suffix=".tmp")
    os.close(fd)
    init_method_base = f"file://{store_path}"
    debug = os.environ.get("DDPPO_LAUNCHER_DEBUG", "0") == "1"
    if debug:
        print(f"[launcher] init_method_base={init_method_base}")

    # Pick a free local TCP port (not used when file:// init_method is passed, but keep
    # env vars set for compatibility if a user overrides --init-method).
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        master_port = s.getsockname()[1]

    procs: List[subprocess.Popen] = []
    try:
        for rank in range(nproc):
            # Start from a sanitized environment to avoid inheriting torchrun/elastic variables
            # that can override rendezvous behavior.
            env = {
                k: v
                for k, v in os.environ.items()
                if not k.startswith("TORCHELASTIC_")
                and not k.startswith("PET_")
                and k
                not in {
                    "RANK",
                    "WORLD_SIZE",
                    "LOCAL_RANK",
                    "LOCAL_WORLD_SIZE",
                    "MASTER_ADDR",
                    "MASTER_PORT",
                }
            }
            env["RANK"] = str(rank)
            env["WORLD_SIZE"] = str(nproc)
            env["LOCAL_RANK"] = str(rank)

            # Force local rendezvous on loopback to avoid hostname/IPv6 resolution issues.
            env["MASTER_ADDR"] = "127.0.0.1"
            env["MASTER_PORT"] = str(master_port)
            env.setdefault("GLOO_SOCKET_IFNAME", "lo0")
            env.setdefault("PYTHONUNBUFFERED", "1")

            cmd = [
                args.python,
                "-m",
                "src.train_ddppo",
                "--backend",
                "gloo",
            ]

            # If the user didn't specify an init method, force the file store.
            train_args = args.train_args
            if train_args and train_args[0] == "--":
                train_args = train_args[1:]

            if "--init-method" not in train_args:
                init_method_rank = f"{init_method_base}?rank={rank}&world_size={nproc}"
                cmd += ["--init-method", init_method_rank]

            cmd += train_args

            if debug:
                print(f"[launcher] rank={rank} cmd={' '.join(cmd)}")

            procs.append(subprocess.Popen(cmd, env=env))

        exit_codes = [p.wait() for p in procs]
        if any(code != 0 for code in exit_codes):
            raise SystemExit(max(exit_codes))
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=2)
            except Exception:
                if p.poll() is None:
                    p.kill()

        try:
            os.remove(store_path)
        except Exception:
            pass


if __name__ == "__main__":
    main()
