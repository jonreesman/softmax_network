#[warn(dead_code)]
use ndarray::{Array, Array2, arr2, Dim, Axis, ArrayView };
use ndarray_stats::QuantileExt;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;


type FloatTensor = Array2<f32>;

struct LayerDense {
    weights: FloatTensor,
    biases: FloatTensor,
    inputs: Option<FloatTensor>
}

impl LayerDense {
    fn new(n_inputs: usize, n_neurons: usize) -> Self {
        LayerDense {
            weights: 0.1 * Array::<f32, _>::random([n_inputs, n_neurons], Uniform::new(-1., 1.)),
            biases: Array::<f32, _>::zeros([1, n_neurons]),
            inputs: None
        }
    }

    fn forward(&mut self, inputs: FloatTensor) -> FloatTensor {
        self.inputs = Some(inputs.clone());
        
        inputs.dot(&self.weights) + self.biases.clone()
    }

    fn backward(self, dvalues: FloatTensor) -> Option<FloatTensor> {
        match self.inputs {
            None => None,
            Some(ins) => {
                let _dweights = ins.t().dot(&dvalues);
                let _dbiases = dvalues.map_axis(Axis(0), |x| x.view().sum());

                let output = ins.dot(&self.weights.t());
                
                Some(output)
            }
        }
    }
}


struct ReLu {
    inputs: FloatTensor,
    outputs: FloatTensor,
    dinputs: FloatTensor
}

impl ReLu {
    fn forward(inputs: FloatTensor) -> (FloatTensor, ReLu) {
        let output = inputs.map(|x| if *x >= 0.0 { *x } else { 0. });
        
        (
            output.clone(), 
            ReLu {
                inputs: inputs.clone(),
                outputs: output,
                dinputs: Array2::zeros(inputs.dim())
            }
        )
    }

    fn backward(&mut self, dvalues: FloatTensor) -> FloatTensor {
        self.dinputs = dvalues.clone();
        for ((x,y),v) in self.dinputs.indexed_iter_mut() {
            if dvalues[(x,y)] < 0. {
                *v = 0.
            } else {
                *v = 1.
            }
        }

        self.dinputs.clone()
    }
}

struct SoftMax {
    outputs: FloatTensor,
    dinputs: FloatTensor
}

impl SoftMax {

    fn forward(inputs: &FloatTensor) -> (FloatTensor, SoftMax) {
        let mut maxes = inputs.map_axis(Axis(1), |x| *x.view().max().unwrap());
        maxes.swap_axes(0, 1);
        let exp_values = (inputs - maxes).mapv(|x| x.exp());
        let mut sums = exp_values.map_axis(Axis(1), |x| x.sum());
        sums.swap_axes(0, 1);
        let outputs = exp_values / sums;
        (   
            outputs.clone(),
            SoftMax {
                outputs,
                dinputs: Array2::zeros(inputs.dim())
            }
        )
    }

    fn backward(&mut self, dvalues: FloatTensor) -> FloatTensor {
        self.dinputs = Array2::zeros(dvalues.dim());

        let mut index: usize = 0;
        for row in self.outputs.axis_iter(Axis(1)) {
            let mut x = row.clone();
            x.swap_axes(0, 1);
            let jacobian_matrix = Array2::from_diag(&x) - x.dot(&x.t());
            let mut axis_mut = self.dinputs.index_axis_mut(Axis(1), index);
            let y_axis = dvalues.index_axis(Axis(1), index);
            axis_mut.assign(&jacobian_matrix.dot(&y_axis));
            index += 1;
        }

        self.dinputs.clone()
    }
}

fn main() {
    let x = arr2(
        & [[1.,2.,3.,2.5],
         [2.0,5.0,-1.0,2.0],
         [-1.5,2.7,3.3,-0.8]]
    );

    let mut layer1 = LayerDense::new(4, 5);
    let mut layer2 = LayerDense::new(5, 2);

    let mut out = layer1.forward(x);
    out = layer2.forward(out);
    let (relu_out, _) = ReLu::forward(out);
    println!("{:?}", relu_out);

    let y = arr2(
        & [[1.,2.,3.,2.5],
         [2.0,5.0,-1.0,2.0],
         [-1.5,2.7,3.3,-0.8]]
    );
    let (outs, _) = SoftMax::forward(&y);
    println!("Softmax");
    println!("{:?}", outs)

}