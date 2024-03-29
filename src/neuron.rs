use uuid::Uuid;

use crate::Value;

pub struct Neuron {
    weights: Vec<Value>,
    bias: Value,
    pub subgraph_id: Option<Uuid>,
}

impl Neuron {
    pub fn new(nin: u32) -> Self {
        let subgraph_id = Some(Uuid::new_v4());

        // Initialize weights with random values between -1 and 1
        let weights: Vec<Value> = (0..nin)
            .map(|_| Value::from(rand::random::<f64>() * 2.0 - 1.0))
            .collect();

        // assign neuron's subgraph to each weight
        for weight in weights.iter() {
            weight.borrow_mut().subgraph_id = subgraph_id;
        }

        // Initialize bias with random value between -1 and 1
        let bias = Value::from(rand::random::<f64>() * 2.0 - 1.0);
        // assign neuron's subgraph to bias
        bias.borrow_mut().subgraph_id = subgraph_id;

        Neuron {
            weights,
            bias,
            subgraph_id: Some(Uuid::new_v4()),
        }
    }

    pub fn forward(&self, inputs: Vec<Value>) -> Value {
        let sum: Value = self
            .weights
            .iter()
            .zip(inputs.iter())
            .map(|(weight, input)| {
                let weighed_input = weight.clone() * input.clone();
                weighed_input.borrow_mut().subgraph_id = self.subgraph_id;
                weighed_input
            })
            .sum();
        sum.borrow_mut().subgraph_id = self.subgraph_id;

        let sum_plus_bias = sum + self.bias.clone();
        sum_plus_bias.borrow_mut().subgraph_id = self.subgraph_id;

        sum_plus_bias
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut params = self.weights.clone();
        params.push(self.bias.clone());

        params
    }
}

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: u32, nout: u32) -> Self {
        let neurons = (0..nout).map(|_| Neuron::new(nin)).collect();

        Layer { neurons }
    }

    pub fn forward(&self, inputs: Vec<Value>) -> Vec<Value> {
        self.neurons
            .iter()
            .map(|neuron| neuron.forward(inputs.clone()))
            .collect()
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.neurons
            .iter()
            .flat_map(|neuron| neuron.parameters())
            .collect()
    }
}

pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(nin: u32, nouts: Vec<u32>) -> Self {
        let mut layers = Vec::new();

        let mut prev_nout = nin;
        for nout in nouts {
            layers.push(Layer::new(prev_nout, nout));
            prev_nout = nout;
        }

        MLP { layers }
    }

    pub fn forward(&self, inputs: Vec<Value>) -> Vec<Value> {
        let mut outputs = inputs;
        for layer in self.layers.iter() {
            outputs = layer.forward(outputs);
        }

        outputs
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }

    pub fn zero_grad(&self) {
        let params = self.parameters();
        for param in params.iter() {
            param.borrow_mut().grad = 0.0;
        }
    }
}
