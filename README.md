# micrograd-rust
 
Building my own Neural networks in Rust. Inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) and [related video](https://youtu.be/VMj-3S1tku0?si=0AJEx-81hEmTKzqf). The code was built piece by piece from the ground up by me. I also took inspiration from [rustygrad](https://github.com/Mathemmagician/rustygrad) when figuring out how to make the Rust borrow checker happy with a DAG.

## Installation
- You must have [graphviz](https://graphviz.org/download/) installed for the graph pngs to be generated.


## Examples

### Creating and visualizing nodes

#### A simple equation
```rust
let a = Value::from(3.0);
let b = Value::from(4.0);
let c = Value::from(5.0);
let d = (a - b) * c;

d.backward();

graph::render_graph(&d).unwrap()
```

![image](https://github.com/sloganking/micrograd-rust/assets/16965931/156dc734-3cdb-4869-9019-5ce252647154)

#### A single neuron with 3 inputs
![graph](https://github.com/sloganking/micrograd-rust/assets/16965931/0a0bb79b-0359-48af-8c6e-8e6427e7c913)

#### A [4,4,1] layer MLP
![graph2](https://github.com/sloganking/micrograd-rust/assets/16965931/9abbda78-ea58-488a-aea0-f686db012abc)
