from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import math
import time
import numpy as np
from compression.checkpoint_io import inventory,source_blocks,empty_destination,copy_sidecars,PackedWriter
from compression.codecs import source_bf16,exact_encode,quant_encode
from decompression.codecs import decode,bf16_float

def compress_checkpoint(checkpoint_path,output_path,profile="exact",workers=4,level=1,reuse_exact=None):
    source=Path(checkpoint_path);rows=inventory(source);out=empty_destination(output_path)
    copy_sidecars(source,out);writer=PackedWriter(out);records=[];started=time.perf_counter()
    reuse_records={};reuse_payloads={}
    gain_used=0
    gain_budget=int(.1995*8.0585*1024**3) if profile=='exact_band10' else None
    def select_band_payload(encoded,stored):
        nonlocal gain_used
        gain=len(stored)-len(encoded)
        if gain>0 and gain_used+gain<=gain_budget:
            gain_used+=gain
            return encoded,'entropy'
        return stored,'deflate_stored'
    if reuse_exact:
        reuse_manifest=json.loads((Path(reuse_exact)/"compression_manifest.json").read_text())
        if reuse_manifest["profile"]!="exact": raise ValueError("Reuse requires an exact BF16 checkpoint")
        reuse_records={r["name"]:r for r in reuse_manifest["tensors"]}
        reuse_payloads={r["name"]:r for r in inventory(reuse_exact)}
        if set(reuse_records)!={r["name"] for r in rows}: raise ValueError("Reuse tensor inventory mismatch")
    def work(raw,dtype,quantized):
        reference=source_bf16(raw,dtype)
        payload=quant_encode(reference,8,128,level) if quantized else exact_encode(reference,level)
        restored=decode(payload)
        if not quantized and not np.array_equal(reference,restored): raise AssertionError("Lossless codec mismatch")
        if quantized:
            x=bf16_float(reference).reshape(-1,128);qmax=127
            scales=np.maximum(np.max(np.abs(x),axis=1,keepdims=True)/qmax,np.nextafter(np.float16(0),np.float16(1))).astype(np.float16).astype(np.float32)
            # Integer conversion canonicalizes rounded negative zero.
            q=np.clip(np.rint(x/scales),-qmax,qmax).astype(np.int16)
            from decompression.codecs import bf16_round
            expected=bf16_round(q.astype(np.float32)*scales).ravel()
            if not np.array_equal(expected,restored): raise AssertionError("Independent quantization mismatch")
        delta=bf16_float(restored).astype(np.float64)-bf16_float(reference).astype(np.float64)
        stored=exact_encode(reference,0,choose=False) if profile=='exact_band10' else None
        return payload,reference.tobytes(),restored.tobytes(),float(delta@delta),float(np.sum(bf16_float(reference).astype(np.float64)**2)),stored
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for ti,row in enumerate(rows):
            name=row["name"]
            rec={"name":name,"shape":row["shape"],"dtype":"BF16","source_dtype":row["dtype"]}
            if "visual" in name.lower():
                rec.update({"storage":"zero","elements":row["elements"]});records.append(rec);continue
            quantized=profile=="ffn_int8" and name.startswith("model.language_model.layers.") and ".mlp." in name and name.endswith(".weight") and len(row["shape"])==2
            if reuse_records and not quantized:
                original=reuse_records[name]
                if original["shape"]!=row["shape"]: raise ValueError("Reuse tensor shape mismatch")
                digest=hashlib.sha256()
                for raw in source_blocks(row): digest.update(source_bf16(raw,row["dtype"]).tobytes())
                if digest.hexdigest()!=original["reference_sha256"]: raise ValueError("Source changed since exact checkpoint")
                selected=json.loads(json.dumps(original))
                source_iter=iter(source_blocks(row)) if profile=='exact_band10' else None
                for block in selected["blocks"]:
                    item=reuse_payloads[block["key"]]
                    with item["path"].open("rb") as f:
                        f.seek(item["offset"]);payload=f.read(item["bytes"])
                    if profile=='exact_band10':
                        ref=source_bf16(next(source_iter),row['dtype'])
                        stored=exact_encode(ref,0,choose=False)
                        payload,coding=select_band_payload(payload,stored)
                        block.update({'payload_bytes':len(payload),'coding':coding})
                    writer.add(block["key"],payload)
                records.append(selected)
                if ti%20==0: print(f"Packed {ti+1}/{len(rows)} tensors; {writer.total/1024**3:.3f} GiB; {time.perf_counter()-started:.0f}s",flush=True)
                continue
            rec.update({"storage":"blocks","quantized":quantized,"blocks":[]})
            reference_hash=hashlib.sha256();decoded_hash=hashlib.sha256();pending=[];se=0.;sr=0.
            def consume(future):
                nonlocal se,sr
                payload,ref,restored,error,energy,stored=future.result()
                coding=None
                if profile=='exact_band10': payload,coding=select_band_payload(payload,stored)
                key=f"packed_{len(records):04d}_{len(rec['blocks']):05d}"
                writer.add(key,payload);reference_hash.update(ref);decoded_hash.update(restored);se+=error;sr+=energy
                rec["blocks"].append({"key":key,"elements":len(ref)//2,"payload_bytes":len(payload),"decoded_sha256":hashlib.sha256(restored).hexdigest()})
                if coding: rec['blocks'][-1]['coding']=coding
            for raw in source_blocks(row):
                pending.append(pool.submit(work,raw,row["dtype"],quantized))
                if len(pending)>=workers: consume(pending.pop(0))
            for future in pending: consume(future)
            rec.update({"reference_sha256":reference_hash.hexdigest(),"decoded_sha256":decoded_hash.hexdigest(),"squared_error":se,"squared_reference":sr})
            records.append(rec)
            if ti%20==0: print(f"Packed {ti+1}/{len(rows)} tensors; {writer.total/1024**3:.3f} GiB; {time.perf_counter()-started:.0f}s",flush=True)
    writer.close()
    manifest={"format":"cs6013_packed_bf16_v1","profile":profile,"output_dtype":"BF16","visual_policy":"zero","source_config_sha256":hashlib.sha256((source/"config.json").read_bytes()).hexdigest(),"tensors":records}
    if profile=='exact_band10':
        manifest['storage_policy']={'exact_entropy_gain_budget_bytes':gain_budget,'selected_gain_bytes':gain_used,'unselected_blocks':'BF16 byte planes in DEFLATE stored blocks at level 0','all_text_bit_exact':True}
    (out/"compression_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    summary={"profile":profile,"text_payload_bytes":writer.total,"compression_percent":100*(1-writer.total/(8.0585*1024**3)),"tensor_count":len(records),"quantized_tensor_count":sum(r.get("quantized",False) for r in records),"bit_exact_text":all(r.get("reference_sha256")==r.get("decoded_sha256") for r in records if r["storage"]!="zero"),"relative_l2_text":math.sqrt(sum(r.get("squared_error",0) for r in records)/sum(r.get("squared_reference",0) for r in records)),"elapsed_seconds":time.perf_counter()-started,"saved_shards":len(writer.paths)}
    if profile=='exact_band10':
        assert 10<summary['compression_percent']<20
        summary['storage_policy']=manifest['storage_policy']
    print(json.dumps(summary,indent=2),flush=True)
    return summary
