"""项目统一命令入口。"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def serve(host: str = "127.0.0.1", port: int = 8000):
    from flask import Flask, send_from_directory

    app = Flask(__name__, static_folder=str(ROOT / "docs"), static_url_path="")

    @app.route("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.route("/<path:path>")
    def static_files(path):
        return send_from_directory(app.static_folder, path)

    app.run(host=host, port=port, debug=False)


def main():
    parser = argparse.ArgumentParser(description="A股技术筛选网站")
    parser.add_argument("command", choices=["daily", "intraday", "site", "all", "serve"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.command == "daily":
        from scripts.update_daily import run
        run()
    elif args.command == "intraday":
        from scripts.update_intraday import run
        run()
    elif args.command == "site":
        from scripts.generate_site import run
        run()
        print("静态网站与 JSON 数据检查完成。运行 python main.py serve 可本地预览。")
    elif args.command == "all":
        from scripts.generate_site import run as generate
        from scripts.update_daily import run as daily
        from scripts.update_intraday import run as intraday
        daily()
        intraday()
        generate()
    else:
        serve(args.host, args.port)


if __name__ == "__main__":
    main()

