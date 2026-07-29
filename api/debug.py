from flask import Flask, jsonify
import os

app = Flask(__name__)

@app.route("/debug")
def debug():
    lib_dir = "/var/task/api/lib"
    return jsonify({
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", "NOT SET"),
        "lib_dir_exists": os.path.isdir(lib_dir),
        "lib_dir_contents": os.listdir(lib_dir) if os.path.isdir(lib_dir) else [],
        "libGL_exists": os.path.exists(os.path.join(lib_dir, "libGL.so.1")),
    })
