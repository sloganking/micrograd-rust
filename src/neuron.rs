use crate::Value;

pub struct Neuron {
    weights: Vec<Value>,
    bias: Value,
}

impl Neuron {
    pub fn new(nin: u32) -> Self {
        // Initialize weights with random values between -1 and 1
        let weights = (0..nin)
            .map(|_| Value::from(rand::random::<f64>() * 2.0 - 1.0))
            .collect();

        // Initialize bias with random value between -1 and 1
        let bias = Value::from(rand::random::<f64>() * 2.0 - 1.0);

        Neuron { weights, bias }
    }

    pub fn forward(&self, inputs: Vec<Value>) -> Value {
        // let mut sum = Value::from(0.0);

        let mut results = Vec::new();
        for (weight, input) in self.weights.iter().zip(inputs.iter()) {
            // let test = weight.clone() * input.clone();
            // sum = sum + test;
            // sum += weight.clone() * input.clone();
            results.push(weight.clone() * input.clone());
        }

        let sum: Value = results.into_iter().sum();

        sum + self.bias.clone()
    }
}
