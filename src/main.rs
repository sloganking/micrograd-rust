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
        Value::from(3.0),
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

    println!("ys: {:?}", ys);

    // generate vec of y predictions
    let preds = xs
        .iter()
        .map(|x| mlp.forward(x.clone())[0].clone())
        .collect::<Vec<_>>();

    println!("ypreds: {:?}", preds);

    // get the loss of the preds
    let losses: Vec<Value> = preds
        .iter()
        .zip(ys.iter())
        .map(|(pred, y)| (pred.clone() - y.clone()).pow(Value::from(2.0)))
        .collect();

    println!("losses: {:?}", losses);

    let loss = losses.into_iter().sum::<Value>();

    loss.backward();

    println!("loss: {:?}", loss);

    let out = loss;

    if graph::render_graph(&out).is_none() {
        println!("Error: Rendering dot file to an image failed. Please ensure that you have graphviz installed (https://graphviz.org/download/), and that the \"dot\" command is runnable from your terminal.");
    }

    let params = mlp.parameters();
    println!("params: {:#?}", params);
    println!("num params: {:?}", params.len());

    let step_size = 0.01;

    for param in params {
        println!("param: {:?}", param);
        let grad = param.borrow().grad;
        param.borrow_mut().data += step_size * -grad;
    }

    // training cycle
    for i in 0..100 {
        mlp.zero_grad();

        let preds = xs
            .iter()
            .map(|x| mlp.forward(x.clone())[0].clone())
            .collect::<Vec<_>>();

        let losses: Vec<Value> = preds
            .iter()
            .zip(ys.iter())
            .map(|(pred, y)| (pred.clone() - y.clone()).pow(Value::from(2.0)))
            .collect();

        let batch_loss = losses.into_iter().sum::<Value>();

        println!("training cycle: {} loss: {:?}", i, batch_loss.borrow().data);

        batch_loss.backward();

        // update all the parameters
        let params = mlp.parameters();
        for param in params {
            let grad = param.borrow().grad;
            param.borrow_mut().data += step_size * -grad;
        }
    }

    // display the final loss
    {
        println!("ys: {:?}", ys);

        // generate vec of y predictions
        let preds = xs
            .iter()
            .map(|x| mlp.forward(x.clone())[0].clone())
            .collect::<Vec<_>>();

        println!("ypreds: {:?}", preds);

        // get the loss of the preds
        let losses: Vec<Value> = preds
            .iter()
            .zip(ys.iter())
            .map(|(pred, y)| (pred.clone() - y.clone()).pow(Value::from(2.0)))
            .collect();

        println!("losses: {:?}", losses);

        let loss = losses.into_iter().sum::<Value>();

        loss.backward();

        println!("loss: {:?}", loss);

        let out = loss;

        if graph::render_graph(&out).is_none() {
            println!("Error: Rendering dot file to an image failed. Please ensure that you have graphviz installed (https://graphviz.org/download/), and that the \"dot\" command is runnable from your terminal.");
        }
    }
}
