#[warn(dead_code)]
use ndarray::{stack, Array1};
use ndarray::{Array, Array2, arr2, Dim, Axis, ArrayView };
use ndarray_stats::QuantileExt;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;


type FloatTensor = Array2<f32>;

struct LayerDense {
    weights: FloatTensor,
    biases: FloatTensor,
    inputs: FloatTensor
}

impl LayerDense {
    fn new(n_inputs: usize, n_neurons: usize) -> Self {
        LayerDense {
            weights: 0.1 * Array::<f32, _>::random([n_inputs, n_neurons], Uniform::new(-1., 1.)),
            biases: Array::<f32, _>::zeros([1, n_neurons]),
            inputs: Array2::zeros([n_inputs, n_neurons])
        }
    }

    fn forward(&mut self, inputs: FloatTensor) -> FloatTensor {
        self.inputs = inputs.clone();
        
        inputs.dot(&self.weights) + self.biases.clone()
    }

    fn backward(self, dvalues: FloatTensor) -> FloatTensor {
        let _dweights = self.inputs.t().dot(&dvalues);
        let _dbiases = dvalues.map_axis(Axis(0), |x| x.view().sum());
        let output = self.inputs.dot(&self.weights.t());
                
        output
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
        // Resulting vector resembles:
        // [1., 2., 3., 4.]
        // But we need to reshape from Nx1 to 1xN
        let maxes_1d = inputs.map_axis(Axis(1), |x| *x.view().max().unwrap());
    
        // First, we must add a dimension, making the output:
        // [[1., 2., 3., 4.]]
        let mut maxes_2d = stack!(Axis(0), maxes_1d);

        // Swapping the axes here makes it resemble:
        // [[1.],
        //  [2.],
        //  [3.],
        //  [4.]]
        maxes_2d.swap_axes(0, 1);
        let exp_values = (inputs - maxes_2d).mapv(|x| x.exp());

        // We must now do with sums what we did with the maxes.
        let sums_1d = exp_values.map_axis(Axis(1), |x| x.sum());
        let mut sums_2d = stack!(Axis(0), sums_1d);
        sums_2d.swap_axes(0, 1);
        let outputs = exp_values / sums_2d;
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

pub trait Loss {
    fn forward(&self, output: Array1<f32>, y: Array1<f32>) -> Array1<f32>;

    fn calculate(&self, output: Array1<f32>, y: Array1<f32>) -> f32 {
        let sample_losses = self.forward(output, y);
        let data_loss = sample_losses.mean().unwrap();

        data_loss
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