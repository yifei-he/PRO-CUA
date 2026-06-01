#!/bin/bash

# Check if the number of nodes and config file are passed as arguments
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <number_of_nodes> <config_file>"
  exit 1
fi

# Get the number of nodes and config file from the command-line arguments
NUM_NODES=$1
CONFIG_FILE=$2

# Generate the list of nodes dynamically
NODES=()
for ((i=0; i<NUM_NODES; i++)); do
  NODES+=("node-$i")
done

# Define the required variables
export FORCE_TORCHRUN=1
export NNODES=${#NODES[@]} # Number of nodes
export MASTER_ADDR=${NODES[0]} # Use the first node as the master
export MASTER_PORT=29501

echo $MASTER_ADDR
echo $MASTER_PORT

TARGET_DIR="/data/data/users/t-yifeihe/cua/LLaMA-Factory"

# Function to run the command on a specific node
run_on_node() {
  local NODE=$1
  local NODE_RANK=$2
  ssh "$NODE" "
    source /data/data/users/t-yifeihe/cua/LLaMA-Factory/llama-factory/bin/activate &&
    : "${WANDB_API_KEY:?Set WANDB_API_KEY before running this script}" && 
    FORCE_TORCHRUN=1 NNODES=$NNODES MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT NODE_RANK=$NODE_RANK llamafactory-cli train $CONFIG_FILE
  " &
}

# Run the command on all nodes
for i in "${!NODES[@]}"; do
  echo "Starting on ${NODES[$i]} with NODE_RANK=$i"
  run_on_node "${NODES[$i]}" "$i"
done

# Wait for all processes to complete
wait

echo "Distributed training completed on all nodes."
