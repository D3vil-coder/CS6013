import argparse
import json
from decompression.unpack_checkpoint import decompress_checkpoint

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--model_name",required=True,help="Reference label only; never downloaded")
    p.add_argument("--checkpoint_path",required=True)
    p.add_argument("--output_path",required=True)
    args=p.parse_args()
    print(json.dumps(decompress_checkpoint(args.checkpoint_path,args.output_path),indent=2))
