# The Configuration Rules

## The Structure part

### 1. How the program parsing it
- read the struct list and then construct iteratively or construct recursively in the parallel layer
### 2. Keywords
+ cnn1, cnn2..
+ full1, full2..
+ parallel_layer1, parallel_layer2...
+ out_layer
+ parallel_out_layer
    
### 3. Simple Structure
```json
    "structure": {
      "struct": ["layer1", "layer2", ...],
      "layer1": {
        "para1": xxx,
        ...
      },
      "layer2": {
        "para1": xxx,
        ...
      }
    }

```
