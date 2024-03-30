# micrograd-rust
 
Building my own Neural networks in Rust. Inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) and [related video](https://youtu.be/VMj-3S1tku0?si=0AJEx-81hEmTKzqf). The code was built piece by piece from the ground up by me. I also took inspiration from [rustygrad](https://github.com/Mathemmagician/rustygrad) when initially figuring out how to make the Rust borrow checker happy with a DAG.

## Installation
- You must have [graphviz](https://graphviz.org/download/) installed for the graph pngs to be generated.

## Examples

### A simple equation
```rust
let a = Value::from(3.0);
let b = Value::from(4.0);
let c = Value::from(5.0);
let d = (a - b) * c;

d.backward();

graph::render_graph(&d).unwrap()
```

![simple_operations](https://github.com/sloganking/micrograd-rust/assets/16965931/156dc734-3cdb-4869-9019-5ce252647154)

### A single neuron with 3 inputs

```rust
let inputs = vec![Value::from(1.0), Value::from(2.0), Value::from(3.0)];
let neuron = neuron::Neuron::new(inputs.len().try_into().unwrap());
let out = neuron.forward(inputs);
out.backward();
graph::render_graph(&out, neuron.get_subgraph_tree().unwrap()).unwrap();
```
![neuron](https://github.com/sloganking/micrograd-rust/assets/16965931/4d6f70fb-33f4-436b-8a20-8abb7af7b278)

### A [4,4,1] layer MLP
```rust
let x_inputs = vec![Value::from(1.0), Value::from(2.0), Value::from(3.0)];
let y_target = Value::from(1.0);
let mlp = neuron::MLP::new(x_inputs.len().try_into().unwrap(), vec![4, 4, 1]);
let preds = mlp.forward(x_inputs);
let pred = preds[0].clone();
let loss = (pred.clone() - y_target.clone()).pow(Value::from(2.0));
loss.backward();
graph::render_graph(&loss, mlp.get_subgraph_tree().unwrap()).unwrap();
```
![mlp](https://github.com/sloganking/micrograd-rust/assets/16965931/a642e9e9-e304-4fae-aea1-1ef2392962b7)
