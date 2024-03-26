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

    let inputs = vec![
        Value::from(1.0),
        Value::from(2.0),
        // Value::from(3.0),
        // Value::from(4.0),
    ];

    let mlp = neuron::MLP::new(inputs.len().try_into().unwrap(), vec![4, 4, 1]);

    // create dataset
    let xs = vec![
        vec![Value::from(2.0), Value::from(3.0), Value::from(-1.0)],
        vec![Value::from(3.0), Value::from(-1.0), Value::from(0.5)],
        vec![Value::from(0.5), Value::from(1.0), Value::from(1.0)],
        vec![Value::from(1.0), Value::from(1.0), Value::from(-1.0)],
    ];

    let ys = vec![
        Value::from(1.0),
        Value::from(-1.0),
        Value::from(-1.0),
        Value::from(1.0),
    ];

    // // see what nn currently things of predictions
    // for (x, y) in xs.iter().zip(ys.iter()) {
    //     let out = mlp.forward(x.clone());
    //     let out = out[0].clone();
    //     let loss = (out.borrow().data - y.borrow().data).powf(2.0);
    //     println!("{:?} -> {:?} (expected: {:?}, loss: {:?})", x, out, y, loss);
    // }

    // generate vec of y predictions
    let preds = xs
        .iter()
        .zip(ys.iter())
        .map(|(x, y)| mlp.forward(x.clone())[0].clone())
        .collect::<Vec<_>>();

    println!("ypreds: {:?}", preds);

    // let loss = xs
    //     .iter()
    //     .zip(ys.iter())
    //     .map(|(x, y)| {
    //         let out = mlp.forward(x.clone());
    //         let out = out[0].clone();
    //         (out - y.clone()).powf(2.0)
    //     })
    //     .sum::<Value>();

    // get the loss of the preds
    let losses: Vec<Value> = preds
        .iter()
        .zip(ys.iter())
        .map(|(pred, y)| (pred.clone() - y.clone()).pow(Value::from(2.0)))
        .collect();

    println!("losses: {:?}", losses);

    let loss = losses.into_iter().sum::<Value>();

    println!("loss: {:?}", loss);

    // loss.backward();

    let out = mlp.forward(inputs);

    println!("{:?}", out);

    let out = out[0].clone();

    if graph::render_graph(&out).is_none() {
        println!("Error: Rendering dot file to an image failed. Please ensure that you have graphviz installed (https://graphviz.org/download/), and that the \"dot\" command is runnable from your terminal.");
    }
}
