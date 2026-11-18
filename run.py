import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="LabelPilot local annotation workbench")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    home = Path(__file__).resolve().parent
    os.environ.setdefault("YOLO_CONFIG_DIR", str(home / "data" / "ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(home / "data" / "matplotlib"))
    import uvicorn
    from labelpilot.server import create_app
    uvicorn.run(create_app(home), host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
