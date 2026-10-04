"""Bounded-memory safetensors IO; no model downloads or model construction."""
from pathlib import Path
import json
import math
import shutil
import struct
import numpy as np

WIDTH = {"BF16":2,"F16":2,"F32":4,"U8":1}
SIDECARS = ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
            "merges.txt", "special_tokens_map.json", "added_tokens.json", "chat_template.jinja",
            "generation_config.json", "preprocessor_config.json", "video_preprocessor_config.json")

def read_header(path):
    with Path(path).open("rb") as f:
        size = struct.unpack("<Q",f.read(8))[0]
        return json.loads(f.read(size)), 8+size

def inventory(root):
    rows = []
    for path in sorted(Path(root).glob("*.safetensors")):
        header,base = read_header(path)
        for name,entry in header.items():
            if name == "__metadata__": continue
            rows.append({"name":name,"shape":entry["shape"],"dtype":entry["dtype"],
                         "elements":math.prod(entry["shape"]),"path":path,
                         "offset":base+entry["data_offsets"][0],
                         "bytes":entry["data_offsets"][1]-entry["data_offsets"][0]})
    if len({r["name"] for r in rows}) != len(rows):
        raise ValueError("Duplicate tensor names in checkpoint")
    if not rows: raise ValueError("No safetensors weights found")
    return sorted(rows,key=lambda r:r["name"])

def source_blocks(row,chunk_elements=1048576):
    with row["path"].open("rb") as f:
        f.seek(row["offset"])
        remaining=row["elements"]
        while remaining:
            n=min(chunk_elements,remaining)
            raw=f.read(n*WIDTH[row["dtype"]])
            if len(raw)!=n*WIDTH[row["dtype"]]: raise ValueError("Truncated tensor")
            yield raw
            remaining-=n

def copy_sidecars(source,destination):
    destination=Path(destination)
    for name in SIDECARS:
        path=Path(source)/name
        if path.is_file(): shutil.copyfile(path,destination/name)
    path=destination/"config.json"
    if not path.is_file(): raise ValueError("config.json is required")
    cfg=json.loads(path.read_text(encoding="utf-8"))
    cfg.pop("quantization_config",None)
    cfg["dtype"]="bfloat16"
    if "torch_dtype" in cfg: cfg["torch_dtype"]="bfloat16"
    if isinstance(cfg.get("text_config"),dict):
        cfg["text_config"]["dtype"]="bfloat16"
        if "torch_dtype" in cfg["text_config"]: cfg["text_config"]["torch_dtype"]="bfloat16"
    path.write_text(json.dumps(cfg,indent=2)+"\n",encoding="utf-8")

def empty_destination(path):
    path=Path(path)
    path.mkdir(parents=True,exist_ok=True)
    if any(path.iterdir()): raise ValueError("Output directory must be empty")
    return path

def write_streaming_safetensors(path,entries,block_provider):
    """entries: name, shape, dtype, bytes; provider yields exact stored bytes."""
    header={"__metadata__":{"format":"pt"}}
    offset=0
    for entry in entries:
        header[entry["name"]]={"dtype":entry["dtype"],"shape":entry["shape"],
                              "data_offsets":[offset,offset+entry["bytes"]]}
        offset+=entry["bytes"]
    encoded=json.dumps(header,separators=(",",":"),ensure_ascii=False).encode("utf-8")
    encoded+=b" "*((-len(encoded))%8)
    path=Path(path)
    partial=path.with_suffix(path.suffix+".partial")
    with partial.open("wb") as f:
        f.write(struct.pack("<Q",len(encoded)));f.write(encoded)
        for entry in entries:
            written=0
            for block in block_provider(entry):
                f.write(block);written+=len(block)
            if written!=entry["bytes"]: raise ValueError(f"Wrong byte count: {entry['name']}")
    partial.replace(path)

class PackedWriter:
    def __init__(self,root,max_bytes=256*1024**2):
        self.root=Path(root);self.max_bytes=max_bytes;self.pending=[];self.nbytes=0
        self.mapping={};self.paths=[];self.total=0
    def add(self,name,payload):
        if self.pending and self.nbytes+len(payload)>self.max_bytes: self.flush()
        self.pending.append((name,payload));self.nbytes+=len(payload);self.total+=len(payload)
    def flush(self):
        if not self.pending: return
        filename=f"model-{len(self.paths)+1:05d}.safetensors"
        entries=[{"name":k,"shape":[len(v)],"dtype":"U8","bytes":len(v)} for k,v in self.pending]
        payloads=dict(self.pending)
        write_streaming_safetensors(self.root/filename,entries,lambda e:[payloads[e["name"]]])
        self.mapping.update({k:filename for k,_ in self.pending});self.paths.append(filename)
        self.pending=[];self.nbytes=0
    def close(self):
        self.flush()
        data={"metadata":{"total_size":self.total},"weight_map":self.mapping}
        (self.root/"model.safetensors.index.json").write_text(json.dumps(data,indent=2),encoding="utf-8")

def shards_for_entries(entries,max_bytes=1536*1024**2):
    current=[];size=0
    for e in entries:
        if current and size+e["bytes"]>max_bytes:
            yield current;current=[];size=0
        current.append(e);size+=e["bytes"]
    if current: yield current
