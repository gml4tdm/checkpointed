# Pypelined - A library for management of nonlinear processing pipelines.

---

## Introduction

The Pypelined library provides a small interface for chaining pipelines of 
different operations. The main functionality of the library is the ability 
to maintain checkpoints for every performed operation.

---

## Example 

```python 
from pypelined import Pipeline, NoopStep

pipeline = Pipeline('example-pipeline')

# A regular input 
input1 = pipeline.add_source(
    NoopStep, name='input-1'
)

# An input whose result is also stored as an output in the file 
# 'input-2-output.txt'
input2 = pipeline.add_source(
    NoopStep, name='input-2', is_sink=True, filename='input-2-output.txt'
)

# A processing step which depends on input1 
node3 = pipeline.add_step(NoopStep, name='node-3')
pipeline.connect(input1, node3)

# A processing step which depends on both input2 and node3
node4 = pipeline.add_step(NoopStep, name='node-4')
pipeline.connect(input2, node4)
pipeline.connect(node3, node4)

# Output step which depends on node4
out5 = pipeline.add_sink(NoopStep, name='out-5', filename='out-5-output.txt')
pipeline.connect(node4, out5)

# Build the execution plan
plan = pipeline.build()

# Configuration information for all the steps.
# Empty because NoopStep does not require configuration.
config = {
    input1: {},
    input2: {},
    node3: {},
    node4: {},
    out5: {}
}

# (optional) setup a logger 
import logging
logger = logging.getLogger('example-pipeline')
logger.addHandler(logging.StreamHandler())

# Run the pipeline
plan.execute(
    # Configuration used to instantiate the steps in the pipeline.
    # Must have one entry for every step.
    config_by_step=config,
    
    # Store `input-2-output.txt' and `out-5-output.txt' 
    # in directory ./out/example-pipeline
    output_directory='out',                
    
    # Store checkpoints for every pipeline step 
    # in directory ./checkpoints/example-pipeline
    checkpoint_directory='checkpoints',
    
    # Pass in a logger object in order to enable logging.
    logger=logger
)
```

