# CS6013 model compression submissions

This repository collects Ansh Savla's CS6013 model-compression coursework submissions. The directories preserve the course submission layout and contain separate compression and reconstruction implementations, configurations and instructions. They are experimental submissions, not a general-purpose inference library.

## Submission index

| Week | Submission | Code and instructions |
| --- | --- | --- |
| 05 | `Compression_20`, Submission 01 | [Group-wise quantization](24b0746/Week05/Compression_20/Submission01/) |
| 05 | `Compression_20`, Submission 02 | [MLP-width pruning](24b0746/Week05/Compression_20/Submission02/) |
| 06 | `Compression_10`, Submission 01 | [Quantization submission](24b0746/Week06/Compression_10/Submission01/) |
| 06 | `Compression_20`, Submission 01 | [Quantization submission](24b0746/Week06/Compression_20/Submission01/) |
| 06 | `Compression_40`, Submission 01 | [Quantization submission](24b0746/Week06/Compression_40/Submission01/) |
| 08 | `Compression_10`, Submission 01 | [Checkpoint packing and reconstruction](24b0746/Week08/Compression_10/Submission01/) |
| 08 | `Compression_20`, Submission 01 | [Checkpoint packing and reconstruction](24b0746/Week08/Compression_20/Submission01/) |
| 08 | `Compression_40`, Submission 01 | [Checkpoint packing and reconstruction](24b0746/Week08/Compression_40/Submission01/) |

The compression numbers are submission category names, not a claim that each implementation achieved exactly that percentage. Refer to the individual submission for its method, dtype details, dependencies and commands.

## Using a submission

Each linked folder has its own README, `compress.py`, `decompress.py` and dependency configuration. Use that folder's instructions rather than assuming one environment or command works across all weeks. The Week 08 entries include a lock file and describe their packed and reconstructed tensor dtypes. A compatible input or packed checkpoint must be supplied separately; model checkpoints are not included in this GitHub repository.

The implementations are coursework experiments. No latency, accuracy, quality-retention or exact compression result is established by this index. Review the relevant submission and independently evaluate the checkpoint and target environment before drawing performance conclusions.

