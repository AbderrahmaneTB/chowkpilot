"""Synthesise narration sentences with Kokoro (offline; Apache-2.0 weights) for tools/make_walkthrough.py.

    python tools/tts_kokoro.py jobs.json        # jobs: [{"text": ..., "path": ...}]

Needs kokoro-onnx and soundfile, and the model files kokoro-v1.0.int8.onnx and voices-v1.0.bin in $KOKORO_DIR
(https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0).
"""

import json
import os
import sys

import soundfile as sf
from kokoro_onnx import Kokoro


def main():
    d = os.environ["KOKORO_DIR"]
    tts = Kokoro(os.path.join(d, "kokoro-v1.0.int8.onnx"), os.path.join(d, "voices-v1.0.bin"))
    voice, speed = os.environ.get("KOKORO_VOICE", "af_heart"), float(os.environ.get("KOKORO_SPEED", "1.0"))
    for job in json.load(open(sys.argv[1], encoding="utf-8")):
        audio, sr = tts.create(job["text"], voice=voice, speed=speed, lang="en-us")
        sf.write(job["path"], audio, sr, subtype="PCM_16")


if __name__ == "__main__":
    main()
