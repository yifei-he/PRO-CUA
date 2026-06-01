# LlamaFactory

## Training
The only command you need to start training is 
```
bash scripts/run_multi_node.sh <node_count> <yaml_file>
```
This will set up all the environments etc automatically across all nodes. Remember to change the wandb account in the file, which is default to be mine.

All the yaml files are in `/examples/train_full`.