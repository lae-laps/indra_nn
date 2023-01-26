/* Indra Artificial Neural Network
 * Rust library for creating neural networks
 * coded by: laelaps
 */

use rand::Rng;

const E_CTE: f64 =  2.7182818284;

#[allow(dead_code)]
struct LayerNeuron {
    bias: f64,
    activation: f64,
    weights: Vec<f64>,
}

#[allow(dead_code)]
impl LayerNeuron {
    fn new() -> LayerNeuron {
        Self {
            bias: rand::thread_rng().gen_range(0.0..=1.0),
            activation: 0.0,
            weights: Vec::new(),
        }
    }
}

#[allow(dead_code)]
struct InputNeuron {
    bias: f64,
    activation: f64,
}

impl InputNeuron {
    fn new() -> InputNeuron {
        Self {
            bias: rand::thread_rng().gen_range(0.0..=1.0),
            activation: 0.0, 
        }
    }
}

impl Copy for InputNeuron {}

impl Clone for InputNeuron {
    fn clone(&self) -> Self {
        *self
    }
}


#[allow(dead_code)]
struct Network {
    learning_rate: f64,
    input_layer_size: usize,
    hidden_layer_size: usize,
    output_layer_size: usize,
    input_layer: Vec<InputNeuron>,
    hidden_layer: Vec<LayerNeuron>,
    output_layer: Vec<LayerNeuron>,
    training_data: Vec<Vec<Vec<f64>>>,
}

impl Network {
    fn new(input_layer_size: usize, hidden_layer_size: usize, output_layer_size: usize, learning_rate: f64) -> Network {
        Self {
            learning_rate,
            input_layer_size,
            hidden_layer_size,
            output_layer_size,
            input_layer: Vec::with_capacity(input_layer_size),
            hidden_layer: Vec::with_capacity(hidden_layer_size),
            output_layer: Vec::with_capacity(output_layer_size),
            training_data: Vec::with_capacity(2),
        }
    }
        
    fn fill_input_layer(&mut self, data: Vec<f64>) -> u8 {
        if data.len() != self.input_layer_size {
            // TODO: implement custom error type
            // for now: 1 = error; 0 = good
            return 1
        }
        for i in data.iter() {
            self.input_layer.push(
            InputNeuron {
                activation: *i, 
                bias: rand::thread_rng().gen_range(0.0..=1.0),
            })
        }
        return 0
    }
        
    fn fordward_propagate(&mut self) {
        // hidden layer activation
        for i in 0..self.hidden_layer_size {
            let mut activation = self.hidden_layer[i].bias;
            for i in 0..self.input_layer_size {
                activation += self.input_layer[i].activation * self.hidden_layer[i].weights[i];
            }
            self.hidden_layer[i].activation = self.compress_value(activation);
        }
         
        // output layer activation
        for i in 0..self.output_layer_size {
            let mut activation = self.output_layer[i].bias; // TODO: check if it is hidden or output
            for i in 0..self.hidden_layer_size {
                activation += self.hidden_layer[i].activation * self.output_layer[i].weights[i];
            }
            self.output_layer[i].activation = self.compress_value(activation);
        }
    }

    fn back_propagate(&mut self, epoch: usize) {
        // compute change in output weights
        let mut output_layer_cost: Vec<f64> = Vec::with_capacity(self.output_layer_size); 
        for j in 0..self.output_layer_size {
            let error = self.training_data[epoch][1][j];
            output_layer_cost[j] = error * self.derivative_gradient_descent(self.output_layer[j].activation);
        }

        // compute change in hidden weights
        let mut hidden_layer_cost: Vec<f64> = Vec::with_capacity(self.hidden_layer_size); 
        for j in 0..self.hidden_layer_size {
            let mut error = 0.0;
            for k in 0..self.output_layer_size {
                error += hidden_layer_cost[k] * self.output_layer[j].weights[k]
            }
            hidden_layer_cost[j] = error * self.derivative_gradient_descent(self.hidden_layer[j].activation);
        }
            
        // apply change in output weights
        for j in 0..self.output_layer_size {
            self.output_layer[j].bias += output_layer_cost[j] * self.learning_rate;
            for k in 0..self.hidden_layer_size {
                self.output_layer[k].weights[j] += self.hidden_layer[k].activation * output_layer_cost[j] * self.learning_rate;
            }
        }
            
        // apply change in hidden weights
        for j in 0..self.hidden_layer_size {
            self.hidden_layer[j].bias += hidden_layer_cost[j] * self.learning_rate;
            for k in 0..self.input_layer_size {
                self.hidden_layer[k].weights[j] += self.training_data[epoch][0][k] * hidden_layer_cost[j] * self.learning_rate;
            }
        }
    }

    fn train(&mut self) {
        for epoch in 0..self.training_data.len() {
            // TODO: shuffle training order
            self.fill_input_layer(self.training_data[epoch][0]);
            self.fordward_propagate();
            self.print_fordward_pass_training_results();
            self.back_propagate(epoch);
        }
    }

    fn print_fordward_pass_training_results(&mut self) {
        /*println!("expected output: {}\nobtained output: {}\nerror: {} -> {}%",
            // expected output
            self.output_layer,
            // self.output_layer - expected output,
            // (self.output_layer - expected output) % 100
        );*/
    }

    fn print_network_structure(&mut self) {
        println!("({}) - ({}) - ({})", self.input_layer_size, self.hidden_layer_size, self.output_layer_size);
    }

    fn derivative_gradient_descent(&self, x: f64) -> f64 {
        return x * (1.0 - x);
    }
        
    fn compress_value(&self, x: f64) -> f64 {
        // TODO: implement more than sigmoid
        return 1.0 / (1.0 + (f64::powf(E_CTE, -x)))
    }
}

fn main() {
    // testing the network
    let mut network = Network::new(2, 2, 1, 0.1);
    network.training_data = vec![
    vec![
        vec![0.0, 0.0],
        vec![0.0],
    ], vec![
        vec![0.0, 1.0],
        vec![1.0],
    ], vec![
        vec![1.0, 0.0],
        vec![1.0],
    ], vec![
        vec![1.0, 1.0],
        vec![0.0],
    ]];
    network.print_network_structure();
    network.train();
}

