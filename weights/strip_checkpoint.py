# for importing options on checkpoint load.
import sys
sys.path.append("/".join(sys.path[0].split("/")[:-1])) 

import torch
import argparse


parser = argparse.ArgumentParser(description="Script for "
                                            "removing training state weighs "
                                            "from a checkpoint.")

parser.add_argument('--heavy_checkpoint_path')
parser.add_argument('--output_checkpoint_path')

args = parser.parse_args()

checkpoint = torch.load(args.heavy_checkpoint_path)

keys_to_store = ["state_dict", 'hparams_name', 'hyper_parameters']

new_checkpoint = {}
for key in keys_to_store:
    new_checkpoint[key] = checkpoint[key] 

torch.save(new_checkpoint, args.output_checkpoint_path)