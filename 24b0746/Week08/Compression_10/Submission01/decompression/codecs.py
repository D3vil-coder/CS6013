"""Decode counted payloads directly into BF16 values."""
import json
import struct
import zlib
import numpy as np

def bf16_float(bits):
    return (np.asarray(bits,dtype=np.uint32)<<16).view(np.float32)


def bf16_round(values):
    u=np.asarray(values,dtype=np.float32).view(np.uint32)
    result=((u+np.uint32(0x7fff)+((u>>16)&1))>>16).astype(np.uint16)
    # Keep NaNs as NaNs even when the payload only occupies discarded bits.
    nan=((u&0x7f800000)==0x7f800000)&((u&0x007fffff)!=0)
    return np.where(nan,((u>>16)|0x40).astype(np.uint16),result).astype(np.uint16)


def unpack(raw,n,width):
    a=np.frombuffer(raw[:n],dtype=np.uint8).astype(np.uint16);offset=n;step=(n+7)//8
    for i in range(8,width):
        a|=np.unpackbits(np.frombuffer(raw[offset:offset+step],dtype=np.uint8),bitorder="little")[:n].astype(np.uint16)<<i
        offset+=step
    if offset!=len(raw): raise ValueError("Invalid packed stream")
    return a


def open_envelope(blob):
    n=struct.unpack("<I",blob[:4])[0];meta=json.loads(blob[4:4+n]);offset=4+n;parts=[]
    if meta.get("version")!=1 or meta.get("output_dtype")!="BF16": raise ValueError("Unsupported codec version")
    for block in meta["blocks"]:
        raw=zlib.decompress(blob[offset:offset+block["encoded"]]);offset+=block["encoded"]
        if len(raw)!=block["raw"]: raise ValueError("Invalid block length")
        parts.append(raw)
    if offset!=len(blob): raise ValueError("Trailing payload bytes")
    return meta,parts


def decode(blob):
    meta,parts=open_envelope(blob);n=meta["elements"];codec=meta["codec"]
    if codec=="uniform":
        width=meta["width"];group=meta["group"];q=unpack(parts[0],n,width).reshape(-1,group)
        magnitude=(q&((1<<(width-1))-1)).astype(np.int16)
        signed=np.where(q>>(width-1),-magnitude,magnitude)
        scale=np.frombuffer(parts[1],dtype="<f2").astype(np.float32).reshape(-1,1)
        return bf16_round(signed.astype(np.float32)*scale).ravel()
    low=np.frombuffer(parts[0],dtype=np.uint8).astype(np.uint16)
    high=np.frombuffer(parts[1],dtype=np.uint8).astype(np.uint16)
    if len(low)!=n or len(high)!=n: raise ValueError("Invalid plane length")
    if codec=="bf16_planes": return low|(high<<8)
    if codec=="bf16_split": return (low&127)|(high<<7)|((low&128)<<8)
    if codec=="fp16_planes_patched":
        result=bf16_round((low|(high<<8)).view(np.float16).astype(np.float32))
        result[np.frombuffer(parts[2],dtype="<u4")]=np.frombuffer(parts[3],dtype="<u2")
        return result
    raise ValueError(f"Unknown codec {codec}")

