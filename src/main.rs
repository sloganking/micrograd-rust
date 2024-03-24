mod graph;
mod neuron;
mod value;
use crate::value::Value;

fn main() {
    // let a = Value::from(3.0);
    // let b = Value::from(4.0);
    // let c = Value::from(5.0);
    // let d = (a - b) * c;

    // d.backward();

    // let neuron = neuron::Neuron::new(3);
    // let inputs = vec![Value::from(1.0), Value::from(2.0), Value::from(3.0)];
    // let out = neuron.forward(inputs);
    // let out = out.tanh();
    // out.backward();

    // let layer = neuron::Layer::new(2, 3);
    // let inputs = vec![Value::from(1.0), Value::from(2.0)];
    // let out = layer.forward(inputs);

    let mlp = neuron::MLP::new(3, vec![4, 4, 1]);

    let inputs = vec![Value::from(1.0), Value::from(2.0), Value::from(3.0)];

    let out = mlp.forward(inputs);

    println!("{:?}", out);

    let out = out[0].clone();

    if graph::render_graph(&out).is_none() {
        println!("Error: Rendering dot file to an image failed. Please ensure that you have graphviz installed (https://graphviz.org/download/), and that the \"dot\" command is runnable from your terminal.");
    }
}
