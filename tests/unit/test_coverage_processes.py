"""Verify measurement and offline policy across actual process boundaries."""
import json
import os
from pathlib import Path
import subprocess
import sys


def test_parent_thread_spawn_and_server_coverage(tmp_path):
    unimported = tmp_path / "namespace" / "unimported.py"
    unimported.parent.mkdir()
    unimported.write_text("def uncalled():\n    return 99\n")
    probe = tmp_path / "probe.py"
    probe.write_text('''import multiprocessing
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.request import urlopen

def worker():
    value = 17
    assert value == 17

def thread_worker():
    thread_value = 23
    assert thread_value == 23

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"server")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        with HTTPServer(("127.0.0.1", 0), Handler) as server:
            print(server.server_port, flush=True)
            server.handle_request()
    else:
        thread = threading.Thread(target=thread_worker)
        thread.start()
        thread.join(10)
        assert not thread.is_alive()
        process = multiprocessing.get_context("spawn").Process(target=worker)
        process.start()
        process.join(10)
        assert process.exitcode == 0
        with subprocess.Popen([sys.executable, __file__, "server"], stdout=subprocess.PIPE, text=True) as child:
            port = int(child.stdout.readline())
            with urlopen(f"http://127.0.0.1:{port}", timeout=10) as response:
                assert response.read() == b"server"
            assert child.wait(timeout=10) == 0
''')
    config = tmp_path / "coverage.ini"
    config.write_text(f"""[run]
branch = true
source = {tmp_path}
concurrency = thread,multiprocessing
patch = subprocess,_exit
parallel = true
sigterm = true
data_file = {tmp_path / '.coverage'}
[report]
include_namespace_packages = true
""")
    env = dict(os.environ, COVERAGE_RCFILE=str(config), COVERAGE_FILE=str(tmp_path / ".coverage"))
    # The nested measurement must start from this configuration, independently
    # of any collector measuring pytest itself.
    for key in list(env):
        if key.startswith("COVERAGE_") and key not in {"COVERAGE_RCFILE", "COVERAGE_FILE"}:
            env.pop(key)
    def coverage(*args):
        result = subprocess.run([sys.executable, "-m", "coverage", *args], env=env,
                                cwd=tmp_path, capture_output=True, text=True, timeout=20)
        assert result.returncode == 0, result.stdout + result.stderr
    coverage("run", str(probe))
    coverage("combine")
    report = tmp_path / "report.json"
    coverage("json", "-o", str(report))
    files = json.loads(report.read_text())["files"]
    unmeasured = next(value for name, value in files.items() if Path(name).name == "unimported.py")
    assert unmeasured["summary"]["num_statements"] == 2
    assert unmeasured["executed_lines"] == []
    assert set(unmeasured["missing_lines"]) == {1, 2}
    measured = next(value for name, value in files.items() if Path(name).name == "probe.py")
    lines = probe.read_text().splitlines()
    for statement in ("value = 17", "thread_value = 23", "self.wfile.write(b\"server\")", "process.start()"):
        number = next(i for i, line in enumerate(lines, 1) if line.strip() == statement)
        assert number in measured["executed_lines"], f"missing process coverage: {statement}"


def test_offline_policy_is_inherited_by_children():
    startup = Path(__file__).parents[1] / "fast_startup"
    env = dict(os.environ, COMFYUI_FAST_OFFLINE="1", PYTHONPATH=str(startup))
    code = '''import socket
for action in (
    lambda: socket.getaddrinfo("example.com", 443),
    lambda: socket.socket().connect(("192.0.2.1", 443)),
    lambda: socket.socket().connect_ex(("192.0.2.1", 443)),
    lambda: socket.socket(type=socket.SOCK_DGRAM).sendto(b"x", ("192.0.2.1", 443)),
):
    try:
        action()
    except RuntimeError as error:
        assert "Fast tests prohibit" in str(error)
    else:
        raise AssertionError("external access was allowed")
assert socket.getaddrinfo("localhost", 80)
'''
    subprocess.run([sys.executable, "-c", code], env=env, check=True,
                   capture_output=True, text=True, timeout=10)
