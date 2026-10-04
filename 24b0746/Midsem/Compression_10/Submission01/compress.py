import argparse
from compression.pack_checkpoint import compress_checkpoint

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--model_name",required=True,help="Reference label only; never downloaded")
    p.add_argument("--checkpoint_path",required=True)
    p.add_argument("--output_path",required=True)
    p.add_argument("--profile",choices=["exact","exact_band10","ffn_int8"],default="exact_band10")
    p.add_argument("--workers",type=int,default=4)
    args=p.parse_args()
    compress_checkpoint(args.checkpoint_path,args.output_path,args.profile,args.workers)
