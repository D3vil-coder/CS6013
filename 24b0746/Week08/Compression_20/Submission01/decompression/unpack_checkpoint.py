from pathlib import Path
import hashlib
import json
import math
import time
from compression.checkpoint_io import inventory,empty_destination,copy_sidecars,write_streaming_safetensors,shards_for_entries
from decompression.codecs import decode

def decompress_checkpoint(checkpoint_path,output_path):
    source=Path(checkpoint_path);manifest=json.loads((source/"compression_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format")!="cs6013_packed_bf16_v1": raise ValueError("Unsupported checkpoint format")
    final_out=empty_destination(output_path)
    out=final_out.with_name(final_out.name+".building")
    marker=out/"codec_build_state.json"
    manifest_sha=hashlib.sha256((source/"compression_manifest.json").read_bytes()).hexdigest()
    if out.exists():
        if not marker.is_file() or json.loads(marker.read_text())["manifest_sha256"]!=manifest_sha:
            raise ValueError("Staging directory belongs to a different checkpoint")
    else:
        out.mkdir()
        marker.write_text(json.dumps({"manifest_sha256":manifest_sha}))
    copy_sidecars(source,out);payloads={r["name"]:r for r in inventory(source)}
    entries=[{**r,"bytes":math.prod(r["shape"])*2} for r in manifest["tensors"]]
    def blocks(entry):
        if entry["storage"]=="zero":
            remaining=entry["bytes"]
            while remaining:
                n=min(2*1048576,remaining);yield bytes(n);remaining-=n
            return
        digest=hashlib.sha256()
        for block in entry["blocks"]:
            row=payloads[block["key"]]
            if row["dtype"]!="U8": raise ValueError("Packed payload must be U8")
            with row["path"].open("rb") as f:
                f.seek(row["offset"]);blob=f.read(row["bytes"])
            restored=decode(blob).astype("<u2",copy=False).tobytes()
            if len(restored)!=block["elements"]*2 or hashlib.sha256(restored).hexdigest()!=block["decoded_sha256"]:
                raise ValueError(f"Payload verification failed: {block['key']}")
            digest.update(restored);yield restored
        if digest.hexdigest()!=entry["decoded_sha256"]: raise ValueError(f"Tensor verification failed: {entry['name']}")
    mapping={};total=0;started=time.perf_counter()
    for si,shard in enumerate(shards_for_entries(entries)):
        name=f"model-{si+1:05d}.safetensors"
        write_streaming_safetensors(out/name,shard,blocks)
        mapping.update({r["name"]:name for r in shard});total+=sum(r["bytes"] for r in shard)
        print(f"Wrote BF16 shard {si+1}; {time.perf_counter()-started:.0f}s",flush=True)
    (out/"model.safetensors.index.json").write_text(json.dumps({"metadata":{"total_size":total},"weight_map":mapping},indent=2),encoding="utf-8")
    # The evaluator treats a config file as an existing model. Expose the output
    # directory only after every weight shard and the final index are complete.
    final_out.rmdir()
    out.rename(final_out)
    (final_out/"codec_build_state.json").unlink()
    return {"output_dtype":"BF16","tensor_count":len(entries),"tensor_bytes":total,"elapsed_seconds":time.perf_counter()-started}
