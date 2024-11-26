#!/bin/bash

# 1: script path
# 2: number of nodes
# 3: host node address
# 4: offser node rank

srun ${1} ${2} ${3} ${4}
