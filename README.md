# micrograd-rust
 
Building my own Neural networks in Rust. Inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) and [related video](https://youtu.be/VMj-3S1tku0?si=0AJEx-81hEmTKzqf). The code was built piece by piece from the ground up by me. But I took inspiration from [rustygrad](https://github.com/Mathemmagician/rustygrad) when figuring out how to make the Rust borrow checker happy with a DAG.

## Installation
- You must have [graphviz](https://graphviz.org/download/) installed for the graph pngs to be generated.


## Examples

### Creating and visualizing nodes

```rust
let a = Value::from(3.0);
let b = Value::from(4.0);
let c = Value::from(5.0);
let d = (a - b) * c;

d.backward();

//render d
```

![image](https://github.com/sloganking/micrograd-rust/assets/16965931/8dc993e0-203a-4f0d-b587-adec58eecc31)
