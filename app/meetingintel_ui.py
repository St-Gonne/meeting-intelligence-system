#!/usr/bin/env python3
"""Small, loopback-only MeetingIntel operator interface.

The server owns at most one subprocess. It never interprets Terminal output,
accepts shell commands, or treats a worker exit as proof of saved artifacts.
"""
from __future__ import annotations

import argparse
import fcntl
import hmac
import http.client
import importlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit, parse_qs


ASSETS = Path(__file__).resolve().parent / "ui"
SOURCE_ID = re.compile(r"[A-Za-z0-9_.:-]{1,180}\Z")
ACTIONS = frozenset({"process", "repair_brief"})
ARTIFACTS = frozenset({"transcript", "report", "brief"})
MAX_BODY = 4096


def _emit(event: str, **fields: Any) -> None:
    try:
        importlib.import_module("meetingintel_learning").emit(event, origin="gui", **fields)
    except Exception:
        # Observation must never change the outcome of a core operation.
        pass


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _private_root(path: Path) -> Path:
    path = path.expanduser().absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("UI runtime directory must not be a symlink")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise ValueError("UI runtime directory must be private to this account")
    return path


def _runtime_root() -> Path:
    return Path(os.environ.get("MI_UI_ROOT", str(Path.home() / "Library/Application Support/MeetingIntel/ui")))


def _existing_session(root: Path, expected_port: int) -> dict[str, Any] | None:
    """Reopen only an authenticated instance holding this runtime's server lock."""
    connection = None
    lockfd = -1
    try:
        metadata = root / "session.json"
        fd = os.open(metadata, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "r", encoding="utf-8") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600 or info.st_size > 4096:
                return None
            session = json.load(stream)
        if not isinstance(session, dict) or set(session) != {"origin", "token", "pid", "instance_id"}:
            return None
        if session["origin"] != f"http://127.0.0.1:{expected_port}" or not 1 <= expected_port <= 65535:
            return None
        if (type(session["pid"]) is not int or session["pid"] < 1
                or not isinstance(session["token"], str) or not re.fullmatch(r"[A-Za-z0-9_-]{32,100}", session["token"])
                or not isinstance(session["instance_id"], str) or not re.fullmatch(r"[a-f0-9]{32}", session["instance_id"])):
            return None
        lockfd = os.open(root / "server.lock", os.O_RDWR | os.O_NOFOLLOW)
        try:
            fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            # An unowned lock means stale metadata, even if the port responds.
            return None
        os.kill(session["pid"], 0)
        connection = http.client.HTTPConnection("127.0.0.1", expected_port, timeout=2)
        connection.request("GET", "/api/session", headers={"X-MI-Token": session["token"]})
        response = connection.getresponse()
        body = response.read(4097)
        if response.status != 200 or len(body) > 4096:
            return None
        identity = json.loads(body)
        if identity != {"pid": session["pid"], "instance_id": session["instance_id"]}:
            return None
        return session
    except (OSError, ValueError, TypeError, http.client.HTTPException):
        return None
    finally:
        if connection is not None:
            connection.close()
        if lockfd != -1:
            os.close(lockfd)


class UIError(Exception):
    def __init__(self, status: int, code: str):
        super().__init__(code)
        self.status, self.code = status, code


class UIState:
    def __init__(self, *, adapter: Any = None, runtime_root: Path | None = None,
                 launcher: Any = subprocess.Popen, clock: Any = time.monotonic):
        self.adapter = adapter or importlib.import_module("meetingintel_actions")
        self.runtime_root = _private_root(runtime_root or _runtime_root())
        self.launcher, self.clock = launcher, clock
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.RLock()
        self.confirmations: dict[str, dict[str, Any]] = {}
        self.offered: set[tuple[str, str]] = set()
        self.process: Any = None
        self.operation: dict[str, Any] | None = None
        record = self.runtime_root / "operation.json"
        if record.exists():
            if record.is_symlink():
                raise ValueError("Unsafe UI operation record")
            try:
                saved = json.loads(record.read_text(encoding="utf-8"))
                if (isinstance(saved, dict) and re.fullmatch(r"[0-9a-f]{32}", saved.get("id", ""))
                        and saved.get("action") in ACTIONS and SOURCE_ID.fullmatch(saved.get("source_id", ""))):
                    self.operation = saved
                    self._refresh_operation()
            except (ValueError, OSError, TypeError):
                # Corrupt operation evidence is unknown, never a completed run.
                self.operation = {"status": "unknown", "outcome": "operation_record_unreadable"}

    def _save_operation(self) -> None:
        if self.operation is not None:
            _atomic_json(self.runtime_root / "operation.json", self.operation)

    def _refresh_operation(self) -> dict[str, Any] | None:
        if not self.operation:
            return None
        operation = self.operation
        operation_id = operation.get("id")
        if not operation_id:
            return dict(operation)
        result_path = self.runtime_root / (operation_id + ".result.json")
        if result_path.is_file() and not result_path.is_symlink():
            try:
                if result_path.stat().st_size > 65536:
                    raise ValueError("Oversized result")
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if not isinstance(result, dict) or type(result.get("ok")) is not bool:
                    raise ValueError("Malformed result")
                category = result.get("outcome", "completed" if result["ok"] else "failed")
                if not isinstance(category, str) or not re.fullmatch(r"[a-zA-Z0-9_.:-]{1,100}", category):
                    category = "completed" if result["ok"] else "failed"
                operation.update(status="finished" if result["ok"] else ("deferred" if category == "deferred" else "failed"), outcome=category)
            except (OSError, ValueError, TypeError):
                operation.update(status="unknown", outcome="result_unreadable")
        elif self.process is not None:
            if self.process.poll() is None:
                operation["status"] = "running"
            else:
                operation.update(status="interrupted", outcome="worker_stopped_without_result")
        elif operation.get("status") in {"running", "starting", "unknown"}:
            pid = operation.get("pid")
            alive = False
            if type(pid) is int and pid > 0:
                try:
                    os.kill(pid, 0)
                    alive = True
                except ProcessLookupError:
                    pass
                except PermissionError:
                    alive = True
            operation.update(status="unknown" if alive else "interrupted",
                             outcome="worker_ownership_unknown" if alive else "worker_stopped_without_result")
        return dict(operation)

    def operation_status(self) -> dict[str, Any] | None:
        with self.lock:
            operation = self._refresh_operation()
            if not operation:
                return None
            return {key: operation[key] for key in ("id", "source_id", "action", "status", "outcome") if key in operation}

    def snapshot(self) -> dict[str, Any]:
        result = dict(self.adapter.snapshot())
        result["operation"] = self.operation_status()
        return result

    def _detail(self, source_id: str) -> dict[str, Any]:
        if not isinstance(source_id, str) or not SOURCE_ID.fullmatch(source_id):
            raise UIError(400, "invalid_source")
        try:
            detail = self.adapter.detail(source_id)
        except (KeyError, ValueError, FileNotFoundError):
            raise UIError(404, "source_unavailable") from None
        if not isinstance(detail, dict) or detail.get("id") != source_id:
            raise UIError(404, "source_unavailable")
        return detail

    @staticmethod
    def _allowed(detail: dict[str, Any], action: str) -> bool:
        return bool(detail.get("can_process")) if action == "process" else bool(detail.get("can_repair_brief"))

    def prepare(self, action: str, source_id: str) -> dict[str, Any]:
        if action not in ACTIONS:
            raise UIError(400, "invalid_action")
        with self.lock:
            detail = self._detail(source_id)
            if not self._allowed(detail, action):
                raise UIError(409, "action_no_longer_available")
            token = secrets.token_urlsafe(24)
            self.confirmations = {key: value for key, value in self.confirmations.items()
                                  if value["expires"] > self.clock()}
            if len(self.confirmations) >= 100:
                raise UIError(429, "too_many_pending_confirmations")
            self.confirmations[token] = {"action": action, "source_id": source_id, "expires": self.clock() + 120}
            _emit("action.selected", source_id=source_id, details={"action": action})
            return {"confirmation_id": token, "action": action, "source": detail}

    def execute(self, confirmation_id: str) -> dict[str, Any]:
        with self.lock:
            prepared = self.confirmations.pop(confirmation_id, None)
            if prepared is None or prepared["expires"] <= self.clock():
                raise UIError(409, "confirmation_expired_or_used")
            current = self._refresh_operation()
            if current and current.get("status") in {"starting", "running", "unknown"}:
                _emit("operation.busy", source_id=prepared["source_id"], details={"action": prepared["action"]})
                raise UIError(409, "operation_busy")
            detail = self._detail(prepared["source_id"])
            if not self._allowed(detail, prepared["action"]):
                raise UIError(409, "action_no_longer_available")
            operation_id = uuid.uuid4().hex
            result_file = self.runtime_root / (operation_id + ".result.json")
            operation = {"id": operation_id, "source_id": prepared["source_id"],
                         "action": prepared["action"], "status": "starting"}
            self.operation = operation
            self._save_operation()
            command = [sys.executable, str(Path(__file__).with_name("meetingintel_actions.py")),
                       "worker", prepared["action"], prepared["source_id"], "--result-file", str(result_file)]
            environment = dict(os.environ, MI_ORIGIN="gui", MI_OPERATION_ID=operation_id)
            try:
                self.process = self.launcher(command, env=environment, stdin=subprocess.DEVNULL,
                                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                             start_new_session=True)
            except OSError:
                operation.update(status="failed", outcome="worker_start_failed")
                self._save_operation()
                raise UIError(503, "worker_start_failed") from None
            operation.update(status="running", pid=self.process.pid)
            self._save_operation()
            _emit("action.confirmed", source_id=prepared["source_id"], operation_id=operation_id,
                  details={"action": prepared["action"]})
            return self.operation_status() or {}

    def cancel_confirmation(self, confirmation_id: str) -> dict[str, Any]:
        with self.lock:
            prepared = self.confirmations.pop(confirmation_id, None)
            if prepared:
                _emit("action.cancelled", source_id=prepared["source_id"], details={"action": prepared["action"]})
            return {"ok": True}

    def artifact(self, source_id: str, kind: str) -> dict[str, str]:
        if kind not in ARTIFACTS:
            raise UIError(400, "invalid_artifact")
        detail = self._detail(source_id)
        if kind not in {item.get("kind") for item in detail.get("artifacts", []) if isinstance(item, dict)}:
            raise UIError(404, "artifact_unavailable")
        try:
            text = self.adapter.artifact(source_id, kind)
        except (ValueError, KeyError, OSError):
            raise UIError(404, "artifact_unavailable") from None
        if not isinstance(text, str):
            raise UIError(404, "artifact_unavailable")
        return {"kind": kind, "text": text}

    def offered_detail(self, source_id: str) -> dict[str, Any]:
        detail = self._detail(source_id)
        with self.lock:
            for action in ACTIONS:
                key = (source_id, action)
                if self._allowed(detail, action) and key not in self.offered:
                    self.offered.add(key)
                    _emit("action.offered", source_id=source_id, details={"action": action})
        return detail

    def learning_status(self) -> dict[str, Any]:
        try:
            return importlib.import_module("meetingintel_learning").status()
        except Exception:
            return {"status": "unavailable", "changes": [], "coverage": {"status": "unknown"}}


class UIServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], state: UIState):
        if address[0] != "127.0.0.1":
            raise ValueError("UI requires exact loopback binding")
        self.state = state
        self.instance_id = uuid.uuid4().hex
        lockfile = state.runtime_root / "server.lock"
        if lockfile.is_symlink():
            raise ValueError("Unsafe UI server lock")
        self.lockfd = os.open(lockfile, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self.lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            super().__init__(address, UIHandler)
        except Exception:
            if self.lockfd != -1:
                os.close(self.lockfd)
            self.lockfd = -1
            raise
        self.host = f"127.0.0.1:{self.server_address[1]}"
        self.origin = "http://" + self.host
        try:
            _atomic_json(state.runtime_root / "session.json", {"origin": self.origin, "token": state.token,
                         "pid": os.getpid(), "instance_id": self.instance_id})
        except Exception:
            self.server_close()
            raise

    def server_close(self) -> None:
        super().server_close()
        metadata = self.state.runtime_root / "session.json"
        try:
            if not metadata.is_symlink():
                session = json.loads(metadata.read_text(encoding="utf-8"))
                if session.get("token") == self.state.token and session.get("instance_id") == self.instance_id:
                    metadata.unlink()
        except (OSError, ValueError, AttributeError):
            pass
        if self.lockfd != -1:
            os.close(self.lockfd)
            self.lockfd = -1


class UIHandler(BaseHTTPRequestHandler):
    server: UIServer

    def log_message(self, *_args: Any) -> None:
        # Request URLs and session tokens must not appear in ordinary logs.
        pass

    def _send(self, status: int, payload: Any, content_type: str = "application/json") -> None:
        body = (json.dumps(payload).encode("utf-8") if content_type == "application/json" else payload)
        self.send_response(status)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'")
        self.end_headers()
        self.wfile.write(body)

    def _check(self, *, authenticated: bool, mutation: bool = False) -> None:
        if self.headers.get("Host") != self.server.host:
            raise UIError(403, "invalid_host")
        origin = self.headers.get("Origin")
        if origin and origin != self.server.origin:
            raise UIError(403, "invalid_origin")
        if mutation and origin != self.server.origin:
            raise UIError(403, "origin_required")
        if authenticated and not hmac.compare_digest(self.headers.get("X-MI-Token", ""), self.server.state.token):
            raise UIError(403, "invalid_session")

    def _body(self, expected: set[str]) -> dict[str, Any]:
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            raise UIError(415, "json_required")
        if self.headers.get("Transfer-Encoding"):
            raise UIError(400, "invalid_encoding")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise UIError(400, "invalid_length") from None
        if length < 2 or length > MAX_BODY:
            raise UIError(413, "invalid_body_size")
        try:
            value = json.loads(self.rfile.read(length))
        except (ValueError, UnicodeError):
            raise UIError(400, "invalid_json") from None
        if not isinstance(value, dict) or set(value) != expected or not all(isinstance(v, str) and len(v) <= 200 for v in value.values()):
            raise UIError(400, "invalid_request_shape")
        return value

    def do_GET(self) -> None:
        try:
            parsed = urlsplit(self.path)
            path = parsed.path
            self._check(authenticated=path.startswith("/api/"))
            assets = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"), "/style.css": ("style.css", "text/css")}
            if path in assets and not parsed.query:
                name, kind = assets[path]
                self._send(200, (ASSETS / name).read_bytes(), kind)
                return
            query = parse_qs(parsed.query, keep_blank_values=True)
            if any(len(values) != 1 for values in query.values()):
                raise UIError(400, "invalid_query")
            if path == "/api/session" and not query:
                response = {"pid": os.getpid(), "instance_id": self.server.instance_id}
            elif path == "/api/snapshot" and not query:
                response = self.server.state.snapshot()
            elif path == "/api/detail" and set(query) == {"id"}:
                response = self.server.state.offered_detail(query["id"][0])
            elif path == "/api/artifact" and set(query) == {"id", "kind"}:
                response = self.server.state.artifact(query["id"][0], query["kind"][0])
            elif path == "/api/learning" and not query:
                response = self.server.state.learning_status()
            else:
                raise UIError(404, "not_found")
            self._send(200, response)
        except UIError as error:
            self._send(error.status, {"error": error.code})
        except Exception:
            self._send(500, {"error": "local_evidence_unavailable"})

    def do_POST(self) -> None:
        try:
            self._check(authenticated=True, mutation=True)
            if self.path == "/api/prepare":
                body = self._body({"action", "source_id"})
                response = self.server.state.prepare(body["action"], body["source_id"])
            elif self.path == "/api/action":
                body = self._body({"confirmation_id"})
                response = self.server.state.execute(body["confirmation_id"])
            elif self.path == "/api/cancel":
                body = self._body({"confirmation_id"})
                response = self.server.state.cancel_confirmation(body["confirmation_id"])
            elif self.path == "/api/handoff":
                body = self._body({"source_id"})
                detail = self.server.state._detail(body["source_id"])
                if not detail.get("handoff"):
                    raise UIError(409, "handoff_unavailable")
                _emit("action.terminal_handoff", source_id=body["source_id"])
                response = {"ok": True}
            elif self.path == "/api/terminal":
                body = self._body({"action"})
                if body["action"] not in {"record", "learning_review"}:
                    raise UIError(400, "invalid_action")
                _emit("action.terminal_handoff", details={"action": body["action"]})
                response = {"ok": True}
            else:
                raise UIError(404, "not_found")
            self._send(200, response)
        except UIError as error:
            self._send(error.status, {"error": error.code})
        except Exception:
            self._send(500, {"error": "operation_unavailable"})


def run_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mi ui", description="Open the private local MeetingIntel inbox")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-browser", action="store_true")
    arguments = parser.parse_args(argv)
    if not 0 <= arguments.port <= 65535:
        parser.error("port must be between 0 and 65535")
    try:
        runtime_root = _private_root(_runtime_root())
        existing = _existing_session(runtime_root, arguments.port)
        if existing:
            url = existing["origin"] + "/#token=" + existing["token"]
            print("MeetingIntel inbox is already running: " + existing["origin"])
            if arguments.no_browser or not webbrowser.open(url):
                print("Open this private session: " + url)
            return 0
        server = UIServer(("127.0.0.1", arguments.port), UIState(runtime_root=runtime_root))
    except (OSError, ValueError) as error:
        print("The local inbox could not start. Check that its port is free and its runtime folder is private.", file=sys.stderr)
        return 1
    url = server.origin + "/#token=" + server.state.token
    if hasattr(server.state.adapter, "reconcile"):
        try:
            server.state.adapter.reconcile()
        except Exception:
            print("Usage reconciliation is unavailable; recording and processing remain separate.")
    print("MeetingIntel local inbox: " + server.origin)
    print("Keep this server running. Browser refresh does not stop processing. Control-C closes the inbox server.")
    if arguments.no_browser:
        print("Open this private session: " + url)
    elif not webbrowser.open(url):
        print("Open this private session: " + url)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nInbox closed. Any started processing continues; reopen mi ui to inspect its outcome.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
