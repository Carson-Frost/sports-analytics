"""Download nflfastR's win-probability model and save it where src/fourth_down/wp.py looks for it.

The model ships in the fastrmodels R package (MIT, Sebastian Carl and Ben Baldwin) as a raw XGBoost
model inside an .rda file. No R is needed: the .rda is xz-compressed, and the model is the run of
bytes after its 4-byte length.
"""

import io
import lzma
import struct
import tarfile
import urllib.request
from pathlib import Path

VERSION = "2.1.0"
URL = f"https://cran.r-project.org/src/contrib/fastrmodels_{VERSION}.tar.gz"
LENGTH = 1219925  # bytes in wp_model_spread, from the package's own tests
OUT = Path(__file__).resolve().parents[1] / "models" / "wp_model_spread.ubj"


def main():
    with urllib.request.urlopen(URL) as r, tarfile.open(fileobj=io.BytesIO(r.read())) as tf:
        rda = tf.extractfile("fastrmodels/data/wp_model_spread.rda").read()
    data = lzma.decompress(rda)
    start = data.find(struct.pack(">i", LENGTH)) + 4
    model = data[start:start + LENGTH]
    if len(model) != LENGTH or not model.startswith(b"{L"):
        raise ValueError("wp_model_spread not found where expected; check the fastrmodels version")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_bytes(model)
    print(f"fastrmodels {VERSION} wp_model_spread -> {OUT}")


if __name__ == "__main__":
    main()
