import json
import threading
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from run_pipeline import PipelineConfig, run_pipeline


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "web"
UPLOAD_DIR = BASE_DIR / "web_uploads"
RESULT_DIR = BASE_DIR / "web_results"
HOST = "127.0.0.1"
PORT = 8000

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

JOBS = {}
JOBS_LOCK = threading.Lock()


def guess_content_type(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".html":
        return "text/html; charset=utf-8"
    if suffix == ".css":
        return "text/css; charset=utf-8"
    if suffix == ".js":
        return "application/javascript; charset=utf-8"
    if suffix == ".json":
        return "application/json; charset=utf-8"
    if suffix == ".mp4":
        return "video/mp4"
    if suffix == ".mov":
        return "video/quicktime"
    if suffix == ".avi":
        return "video/x-msvideo"
    return "application/octet-stream"


def json_bytes(payload):
    return json.dumps(payload).encode("utf-8")


def parse_multipart_video(headers, rfile):
    content_type = headers.get("Content-Type", "")
    if "multipart/form-data" not in content_type or "boundary=" not in content_type:
        raise ValueError("Expected multipart/form-data upload.")

    content_length = int(headers.get("Content-Length", "0"))
    if content_length <= 0:
        raise ValueError("Upload body is empty.")

    boundary = content_type.split("boundary=", 1)[1].strip().strip('"')
    raw_body = rfile.read(content_length)
    boundary_bytes = f"--{boundary}".encode("utf-8")
    parts = raw_body.split(boundary_bytes)

    for part in parts:
        part = part.strip()
        if not part or part == b"--":
            continue

        if part.startswith(b"\r\n"):
            part = part[2:]
        if part.endswith(b"--"):
            part = part[:-2]
        if part.endswith(b"\r\n"):
            part = part[:-2]

        header_blob, separator, body = part.partition(b"\r\n\r\n")
        if not separator:
            continue

        header_text = header_blob.decode("utf-8", errors="ignore")
        if 'name="video"' not in header_text:
            continue

        filename = "upload.mp4"
        for line in header_text.split("\r\n"):
            if "filename=" in line:
                filename = line.split("filename=", 1)[1].strip().strip('"')
                break

        return Path(filename).name, body

    raise ValueError("Missing file field named 'video'.")


def update_job(job_id, **changes):
    with JOBS_LOCK:
        JOBS[job_id].update(changes)


def build_video_urls(job_id, input_path, output_path):
    return {
        "input_video_url": f"/media/uploads/{input_path.name}",
        "output_video_url": f"/media/results/{output_path.name}" if output_path else None,
        "job_url": f"/api/jobs/{job_id}",
    }


def process_job(job_id, input_path: Path):
    output_path = RESULT_DIR / f"{job_id}_processed.mp4"
    update_job(
        job_id,
        status="running",
        message="Loading models and starting video analysis.",
        output_path=str(output_path),
    )

    try:
        def progress_callback(progress):
            total_frames = progress["total_frames"]
            frame_index = progress["frame_index"]
            percent = int((frame_index / total_frames) * 100) if total_frames else 0
            update_job(
                job_id,
                status="running",
                frame_index=frame_index,
                total_frames=total_frames,
                processed_frames=progress["processed_frames"],
                progress_percent=percent,
                latest_status_lines=progress["status_lines"],
                event_counts=progress["event_counts"],
                message=f"Processing frame {frame_index}" if frame_index else "Processing video.",
            )

        result = run_pipeline(
            PipelineConfig(
                source=str(input_path),
                save_output=output_path,
                show=False,
                frame_stride=2,
                imgsz=640,
                accident_imgsz=224,
                device="0",
            ),
            progress_callback=progress_callback,
        )

        update_job(
            job_id,
            status="completed",
            progress_percent=100,
            message="Video processing completed.",
            result=result,
            event_counts=result["event_counts"],
            **build_video_urls(job_id, input_path, output_path),
        )
    except Exception as exc:
        update_job(
            job_id,
            status="failed",
            message=str(exc),
            error_trace=traceback.format_exc(),
        )


class AppHandler(BaseHTTPRequestHandler):
    server_version = "AISurveillanceHTTP/1.0"

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            self.serve_file(STATIC_DIR / "index.html")
            return
        if path.startswith("/static/"):
            self.serve_file(STATIC_DIR / path.removeprefix("/static/"))
            return
        if path.startswith("/media/uploads/"):
            self.serve_file(UPLOAD_DIR / path.removeprefix("/media/uploads/"))
            return
        if path.startswith("/media/results/"):
            self.serve_file(RESULT_DIR / path.removeprefix("/media/results/"))
            return
        if path.startswith("/api/jobs/"):
            self.handle_job_status(path)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Route not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/process":
            self.handle_process_upload()
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Route not found")

    def handle_job_status(self, path):
        job_id = path.removeprefix("/api/jobs/").strip("/")
        with JOBS_LOCK:
            job = JOBS.get(job_id)

        if not job:
            self.send_json({"error": "Job not found"}, status=HTTPStatus.NOT_FOUND)
            return

        self.send_json(job)

    def handle_process_upload(self):
        try:
            original_name, file_bytes = parse_multipart_video(self.headers, self.rfile)
        except ValueError as exc:
            self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        job_id = uuid.uuid4().hex[:12]
        stored_name = f"{job_id}_{original_name}"
        upload_path = UPLOAD_DIR / stored_name

        with upload_path.open("wb") as target:
            target.write(file_bytes)

        with JOBS_LOCK:
            JOBS[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "message": "Upload received. Waiting to start processing.",
                "input_name": original_name,
                "input_path": str(upload_path),
                "frame_index": 0,
                "total_frames": 0,
                "processed_frames": 0,
                "progress_percent": 0,
                "latest_status_lines": [],
                "event_counts": {},
            }

        worker = threading.Thread(target=process_job, args=(job_id, upload_path), daemon=True)
        worker.start()

        response = {
            "job_id": job_id,
            "status": "queued",
            "message": "Upload successful. Processing started.",
            "job_url": f"/api/jobs/{job_id}",
        }
        self.send_json(response, status=HTTPStatus.ACCEPTED)

    def serve_file(self, path: Path):
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return

        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", guess_content_type(path))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload, status=HTTPStatus.OK):
        body = json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    server = ThreadingHTTPServer((HOST, PORT), AppHandler)
    print(f"AI Surveillance web app running at http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
