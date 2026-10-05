"""Encode counted byte payloads; reconstruction algorithms live in decompression."""
import json
import struct
import zlib
import numpy as np
from decompression.codecs import bf16_float,bf16_round

def source_bf16(raw,dtype):
    if dtype=="BF16": return np.frombuffer(raw,dtype="<u2")
    if dtype=="F32": return bf16_round(np.frombuffer(raw,dtype="<f4"))
    if dtype=="F16": return bf16_round(np.frombuffer(raw,dtype="<f2").astype(np.float32))
    raise ValueError(f"Unsupported source dtype {dtype}")


def pack(codes,width):
    a=np.asarray(codes,dtype=np.uint16).ravel()
    return (a&255).astype(np.uint8).tobytes()+b"".join(
        np.packbits(((a>>i)&1).astype(np.uint8),bitorder="little").tobytes() for i in range(8,width))


def envelope(meta,parts,level=1):
    blocks=[zlib.compress(p,level) for p in parts]
    meta={"version":1,"output_dtype":"BF16",**meta,"blocks":[
        {"raw":len(p),"encoded":len(b)} for p,b in zip(parts,blocks)]}
    h=json.dumps(meta,sort_keys=True,separators=(",",":")).encode()
    return struct.pack("<I",len(h))+h+b"".join(blocks)


def exact_encode(bits,level=1,choose=True):
    b=np.asarray(bits,dtype=np.uint16).ravel()
    candidates=[envelope({"codec":"bf16_planes","elements":len(b)},
        [(b&255).astype(np.uint8).tobytes(),(b>>8).astype(np.uint8).tobytes()],level)]
    if choose:
        candidates.append(envelope({"codec":"bf16_split","elements":len(b)},
            [((b&127)|((b>>8)&128)).astype(np.uint8).tobytes(),((b>>7)&255).astype(np.uint8).tobytes()],level))
        # FP16 planes can compress better; preserve every BF16 exception explicitly.
        with np.errstate(over="ignore",invalid="ignore"):
            f16=bf16_float(b).astype(np.float16)
        converted=bf16_round(f16.astype(np.float32))
        idx=np.flatnonzero(converted!=b).astype("<u4")
        f=f16.view(np.uint16)
        candidates.append(envelope({"codec":"fp16_planes_patched","elements":len(b),"exceptions":len(idx)},
            [(f&255).astype(np.uint8).tobytes(),(f>>8).astype(np.uint8).tobytes(),idx.tobytes(),b[idx].tobytes()],level))
    return min(candidates,key=len)


def quant_encode(bits,width=8,group=128,level=1,optimize=False):
    if not 8<=width<=12: raise ValueError("Supported precision: INT8 to INT12")
    b=np.asarray(bits,dtype=np.uint16).ravel()
    if len(b)%group: raise ValueError("Group length must divide block")
    x=bf16_float(b).reshape(-1,group)
    limit=2**(width-1)-1
    scale=np.maximum(np.max(np.abs(x),axis=1,keepdims=True)/limit,
                     np.nextafter(np.float16(0),np.float16(1))).astype(np.float16)
    if optimize:
        original=scale.copy();best=np.full(scale.shape,np.inf,dtype=np.float32)
        for factor in [1.,.97,.99,1.01,1.03,1.05,.95]:
            trial=np.maximum(original.astype(np.float32)*factor,np.nextafter(np.float16(0),np.float16(1))).astype(np.float16)
            tq=np.clip(np.rint(x/trial.astype(np.float32)),-limit,limit).astype(np.int16)
            loaded=bf16_float(bf16_round(tq.astype(np.float32)*trial.astype(np.float32)))
            error=np.mean((loaded-x)**2,axis=1,keepdims=True)
            use=error<best;scale=np.where(use,trial,scale);best=np.minimum(best,error)
    signed=np.clip(np.rint(x/scale.astype(np.float32)),-limit,limit).astype(np.int16)
    codes=np.abs(signed).astype(np.uint16)|((signed<0).astype(np.uint16)<<(width-1))
    extra={"optimized_scale":True} if optimize else {}
    return envelope({"codec":"uniform","elements":len(b),"width":width,"group":group,**extra},
                    [pack(codes,width),scale.tobytes()],level)

