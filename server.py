from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.audit_log import AuditLog
from src.data_repository import DataRepository
from src.optimizer import DispatchOptimizer
from src.policy_engine import PolicyEngine
from src.risk_engine import RiskEngine


class PlatformServices:
    def __init__(self, root: Path):
        data_file = root / "data" / "platform_data.json"
        if not data_file.exists():
            raise FileNotFoundError(
                "缺少 data/platform_data.json，请先运行 "
                "python scripts/prepare_demo_data.py --source <大模型构建及管理目录>"
            )
        self.root = root
        self.repository = DataRepository(data_file)
        snapshot = self.repository.snapshot()
        self.risk = RiskEngine(root / "models", snapshot["default_features"])
        self.policy = PolicyEngine(snapshot["policy_library"])
        self.optimizer = DispatchOptimizer()
        self.audit = AuditLog(root / "runtime")


class PlatformHandler(BaseHTTPRequestHandler):
    server_version = "RoadRiskPlatform/1.0"

    @property
    def services(self) -> PlatformServices:
        return self.server.services  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/health":
                self._json(
                    {
                        "status": "ok",
                        "model": self.services.risk.status,
                        "data": self.services.repository.snapshot()["snapshot"],
                    }
                )
                return
            if parsed.path == "/api/bootstrap":
                payload = self.services.repository.overview()
                payload["model_status"] = self.services.risk.status
                payload["audit"] = self.services.audit.list(limit=10)
                self._json(payload)
                return
            if parsed.path == "/api/watchlist":
                query = parse_qs(parsed.query)
                rows = self.services.repository.watchlist(
                    road=query.get("road", ["all"])[0],
                    direction=query.get("direction", ["all"])[0],
                    level=query.get("level", ["all"])[0],
                    query=query.get("q", [""])[0],
                )
                self._json({"count": len(rows), "rows": rows})
                return
            if parsed.path == "/api/governance":
                payload = self.services.repository.governance()
                payload["model_status"] = self.services.risk.status
                self._json(payload)
                return
            if parsed.path == "/api/audit":
                self._json({"rows": self.services.audit.list()})
                return
            if parsed.path.startswith("/api/cases/"):
                event_id = parsed.path.split("/", 3)[-1]
                self._json(self.services.repository.case(event_id))
                return
            if parsed.path.startswith("/api/"):
                self._error(HTTPStatus.NOT_FOUND, "接口不存在")
                return
            self._static(parsed.path)
        except KeyError as exc:
            self._error(HTTPStatus.NOT_FOUND, str(exc))
        except Exception as exc:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            payload = self._body()
            if parsed.path == "/api/risk/evaluate":
                self._json(self.services.risk.evaluate(payload))
                return
            if parsed.path == "/api/llm/review":
                self._json(self.services.policy.review(payload))
                return
            if parsed.path == "/api/dispatch/optimize":
                rows = self.services.repository.watchlist(
                    road=str(payload.get("road", "all")),
                    direction=str(payload.get("direction", "all")),
                    level=str(payload.get("level", "all")),
                )
                self._json(self.services.optimizer.allocate(rows, payload))
                return
            if parsed.path == "/api/audit":
                self._json(self.services.audit.append(payload), status=HTTPStatus.CREATED)
                return
            self._error(HTTPStatus.NOT_FOUND, "接口不存在")
        except (ValueError, json.JSONDecodeError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except RuntimeError as exc:
            self._error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
        except Exception as exc:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        if length > 1_000_000:
            raise ValueError("请求体超过 1 MB 限制")
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("请求体必须为 JSON 对象")
        return value

    def _json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json({"error": message}, status=status)

    def _static(self, request_path: str) -> None:
        web_root = (self.services.root / "web").resolve()
        relative = "index.html" if request_path in {"", "/"} else request_path.lstrip("/")
        candidate = (web_root / relative).resolve()
        if web_root not in candidate.parents and candidate != web_root:
            self._error(HTTPStatus.FORBIDDEN, "禁止访问该路径")
            return
        if not candidate.is_file():
            candidate = web_root / "index.html"
        body = candidate.read_bytes()
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in {"application/javascript", "application/json"}:
            content_type += "; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {self.address_string()} {format % args}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="大模型道路路域风险监管平台")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认 127.0.0.1")
    parser.add_argument("--port", type=int, default=8765, help="监听端口，默认 8765")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    services = PlatformServices(ROOT)
    server = ThreadingHTTPServer((args.host, args.port), PlatformHandler)
    server.services = services  # type: ignore[attr-defined]
    print(f"大模型道路路域风险监管平台已启动：http://{args.host}:{args.port}")
    print("当前为历史因果回放与人工复核模式，不会向生产设备下发指令。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止。")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

