"""One owner captures each operation, including child output and failure traces."""
from contextlib import contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import shlex
import sys
import threading
import time
import traceback
import uuid

from .common import atomic_json


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def sanitized(value):
    if isinstance(value, str):
        token = os.environ.get("MODELSCOPE_API_TOKEN")
        return value.replace(token, "[REDACTED]") if token else value
    if isinstance(value, dict):
        return {key: sanitized(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitized(item) for item in value]
    return value


class Transcript:
    def __init__(self, target, console_timestamps=True):
        self.target = target
        self.console_timestamps = console_timestamps
        self.mutex = threading.RLock()

    def line(self, stream, value):
        token = os.environ.get("MODELSCOPE_API_TOKEN")
        if token:
            value = value.replace(token, "[REDACTED]")
        line = f"{utc()} {value}\n"
        with self.mutex:
            stream.write(line if self.console_timestamps else value + "\n")
            stream.flush()
            self.target.write(line)
            self.target.flush()


class Tee:
    def __init__(self, original, transcript):
        self.original, self.transcript, self.pending = original, transcript, ""
        self.encoding = "utf-8"

    def write(self, text):
        self.pending += text.replace("\r\n", "\n").replace("\r", "\n")
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            if line:
                self.transcript.line(self.original, line)
        return len(text)

    def flush(self):
        self.original.flush()
        self.transcript.target.flush()

    def finish(self):
        if self.pending:
            self.transcript.line(self.original, self.pending)
            self.pending = ""

    def isatty(self):
        return False


class Operation:
    def __init__(self, directory, name, arguments=(), state_dir=None, *, console_timestamps=True):
        self.directory = Path(directory)
        self.name = name
        self.state_dir = Path(state_dir) if state_dir else None
        self.id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
        self.log = self.directory / f"{name}-{self.id}.log"
        self.summary = self.log.with_suffix(".json")
        self.data = {"operation_id": self.id, "name": name, "status": "RUNNING", "started_at": utc(),
                     "log": str(self.log), "phases": []}
        self.arguments = list(arguments)
        self.console_timestamps = console_timestamps

    def save(self):
        atomic_json(self.summary, sanitized(self.data))
        if self.state_dir:
            atomic_json(self.state_dir / "latest-operation.json", sanitized(self.data))

    @contextmanager
    def phase(self, name):
        phase = {"phase": name, "status": "RUNNING", "started_at": utc()}
        self.data["phases"].append(phase)
        self.save()
        started = time.monotonic()
        print(f"[START] {name}", flush=True)
        try:
            yield
        except BaseException as exc:
            phase.update(status="FAILED", error=str(exc) or type(exc).__name__)
            raise
        else:
            phase["status"] = "PASSED"
        finally:
            phase.update(finished_at=utc(), seconds=round(time.monotonic() - started, 3))
            print(f"[{'PASS' if phase['status'] == 'PASSED' else 'FAIL'}] {name} elapsed={phase['seconds']:.3f}s", flush=True)
            self.save()

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stream = self.log.open("w", encoding="utf-8", buffering=1)
        self.old_out, self.old_err = sys.stdout, sys.stderr
        sink = Transcript(self.stream, self.console_timestamps)
        sys.stdout, sys.stderr = Tee(self.old_out, sink), Tee(self.old_err, sink)
        self.previous_log = os.environ.get("PIPELINE_LOG_PATH")
        os.environ["PIPELINE_LOG_PATH"] = str(self.log)
        self.started = time.monotonic()
        print(f"[START] Operation {self.name}; id={self.id}")
        print(f"[INFO] Command: {shlex.join(self.arguments)}")
        print(f"[INFO] Log: {self.log}")
        print(f"[INFO] Summary: {self.summary}")
        self.save()
        return self

    def __exit__(self, kind, exc, tb):
        try:
            self.data.update(status="FAILED" if exc else "PASSED", finished_at=utc(),
                             seconds=round(time.monotonic() - self.started, 3))
            if exc:
                self.data["error"] = str(exc) or kind.__name__
                # Redact the structured summary as well as the transcript.
                token = os.environ.get("MODELSCOPE_API_TOKEN")
                if token:
                    self.data["error"] = self.data["error"].replace(token, "[REDACTED]")
                    for phase in self.data["phases"]:
                        if "error" in phase:
                            phase["error"] = phase["error"].replace(token, "[REDACTED]")
                traceback.print_exception(kind, exc, tb)
            print(f"[{'FAIL' if exc else 'PASS'}] Operation {self.name}; elapsed={self.data['seconds']:.3f}s")
            print(f"[INFO] Log: {self.log}")
            self.save()
        finally:
            sys.stdout.finish()
            sys.stderr.finish()
            sys.stdout, sys.stderr = self.old_out, self.old_err
            self.stream.close()
            if self.previous_log is None:
                os.environ.pop("PIPELINE_LOG_PATH", None)
            else:
                os.environ["PIPELINE_LOG_PATH"] = self.previous_log
        return False
