# -*- coding: utf-8 -*-

'''
A basic attempt to create a neural network. Just for fun, really.
Based on Michael Nielsen's book: www.neuralnetworksanddeeplearning.com
'''

import numpy as np
from typing import Callable, Union, Optional, List, Tuple

## Basic computations

# acitvation function & derivative

def sigmoid(
    x: Union[np.ndarray, float],
) -> Union[np.ndarray, float]:

    return (1 + np.exp(-x))**-1

def sigmoid_deriv(
    x: Union[np.ndarray, float],
) -> Union[np.ndarray, float]:

    return sigmoid(x) * (1 - sigmoid(x))

def linear(
    x: Union[np.ndarray, float],
) -> Union[np.ndarray, float]:
    
    return x

def linear_deriv(
    x: Union[np.ndarray, float],
) -> Union[np.ndarray, float]:
    
    return 1.0

# objective function & derivative

def L2(
    y_pred: Union[np.ndarray, float],
    y_true: Union[np.ndarray, float],
) -> float:
    
    return 0.5*np.sum((y_pred - y_true)**2)

def L2_deriv(
    y_pred: Union[np.ndarray, float],
    y_true: Union[np.ndarray, float],
) -> np.ndarray:
    
    return (y_pred - y_true)

class Layer:

    def __init__(
        self,
        n_neurons: int,
        activation_fn: Union[Callable, str] = 'sigmoid',
        n_inputs: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        
        # prepare activation as function
        if type(activation_fn) == str:
            if activation_fn == 'sigmoid':
                activation_fn = sigmoid
            elif activation_fn == 'linear':
                activation_fn = linear
            else:
                raise ValueError(
                    'activation_fn as string only recognises `sigmoid`'
                )
        
        self.n_neurons = n_neurons
        self.activation_fn = activation_fn
        if n_inputs: self.n_inputs = n_inputs
        
        # set derivative of activation fn
        if self.activation_fn == sigmoid:
            self.activation_deriv = sigmoid_deriv
        elif self.activation_fn == linear:
            self.activation_deriv = linear_deriv
        
        # set random state for the layer (numpy default_rng)
        if random_seed:
            self.random_generator = np.random.default_rng(random_seed)
        else:
            self.random_generator = np.random.default_rng()
        
    def initialise_weights_and_biases(
        self,
        previous_layer: Optional['Layer']=None,
    ) -> None:
        
        '''
        Initialise weights & biases with normally distributed values (mu=0, sigma=1).
        Normally distributed works best because of the sigmoid function. If for instance
        a flat distribution were chosen, lots of values would be close to 0 and 1, where
        sigmoid is nearly flat which means slow (or even no) learning (derivative ~0).
        '''
        
        # bit ugly, makes sure we know how many connections prev-current layer we need
        if not previous_layer:
            # print('got no previous layer')
            assert hasattr(self, 'n_inputs'), "The first layer needs n_inputs specified."
        else:
            self.n_inputs = previous_layer.n_neurons
        
    
        self.weights = self.random_generator.standard_normal((self.n_neurons, self.n_inputs))
        self.biases = self.random_generator.standard_normal((self.n_neurons, 1))
        

class Network:
    
    def __init__(
        self,
        name: Optional[str] = None,
    ) -> None:
        
        self.layers = []
        self.n_layers = 0
        if name: self.name = name
        
    def __repr__(
        self
    ) -> str:
        
        name=self.name if hasattr(self, 'name') else ''
        str = f'Network {name} with {self.n_layers} layers:\n'
        for _i, layer in enumerate(self.layers):
            str += f'  Layer {_i}: {layer.n_neurons} neurons,'\
                +  f' activation function: {layer.activation_fn}\n'
                
        return str
        
    def add(
        self,
        layer: Layer,
    ) -> None:
        
        print(f'initialising layer {self.n_layers}')
        
        assert isinstance(layer, Layer), f"You have to add a Layer, not a {type(layer)}"
        
        # If first layer: check if it has n_inputs supplied
        if self.n_layers == 0:
            assert hasattr(layer, 'n_inputs'), "The first layer must have n_inputs specified"

        # Add the layer to the list
        self.layers.append(layer)
        self.n_layers += 1
        
        # Initialise weights & biases (and connect w/ previous layer)
        if self.n_layers > 1:
            prev = self.n_layers-2 # python index starts at 0: most recent layer is -1, so prev = -2
            print(f'previous layer: {prev}')
            layer.initialise_weights_and_biases(self.layers[prev])
        else:
            layer.initialise_weights_and_biases()
            
    def reset_weights_and_biases(
        self
    ) -> None:
        
        for layer in self.layers:
            layer.initialise_weights_and_biases()
            
    def compute_forward(
        self,
        X: np.ndarray,
        verbose: bool = False,
    ) -> np.ndarray:
        
        '''Do a forward computation of the network outputs'''
        
        X = np.asarray(X)
        assert X.shape[1] == self.layers[0].n_inputs, "2nd input shape should be # inputs of first layer."
        if verbose: print(f'Computing forward for {X.shape[0]} samples')
        
        # X starts off as network input, turns into each next layer's output.
        # The function acting on the weighted values depends on the layer.
        # For now, this is just the sigmoid.
        
        # # attempting to vectorise -- failure so far.
        # X = np.reshape(X, (X.shape[1], 1, X.shape[0]))
        # # X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        # print(f'X new shape: {X.shape}')
        # for layer in self.layers:
        #     print(f'shapes: weights: {layer.weights.shape} - X: {X.shape} - biases: {layer.biases.shape}')
        #     X = layer.activation_fn(np.dot(layer.weights, X) + layer.biases)
        # return X
        
        # TODO This is REALLY stupid to do it sample by sample
        ys = []
        for a in X:
            # print(a.shape)
            a = a.reshape(a.shape[0],1)
            for layer in self.layers:
                if verbose: print(f'shapes: weights: {layer.weights.shape} - a: {a.shape} - biases: {layer.biases.shape}')
                a = layer.activation_fn(np.dot(layer.weights, a) + layer.biases)
            # print(a.shape)
            ys.append(a)
            
        return np.asarray(ys).reshape(len(ys))
        # return ys
    
    # def eval_objective(
    #     self,
    #     X: np.ndarray,
    #     objective_fn: Union[str, function] = L2,
    # ) -> float:
        
    #     pass
    
    def count_correct_predictions(
        self,
        test_data: Tuple[np.ndarray, np.ndarray],
    ) -> float:
        
        '''
        Return the number of tests where the network was 'right'.
        This is assumed to be the neuron that has the highest value.
        '''
        
        # print(test_data)
        (X_test, y_test) = test_data
        y_pred = [np.argmax(self.compute_forward(Xt)) for Xt in X_test]
        # check whether the predicted value is the actual value & sum
        n_correct = sum(int(yp==yt) for yp, yt in zip(y_pred, y_test))
        
        return n_correct
    
        # test_results = [
        #     (np.argmax(self.compute_forward(Xt)), yt)
        #     for (Xt, yt) in zip(test_data)]
        # # check wherever the predicted value is the actual value
        # return sum(int(yp==yt) for yp,yt in test_results)

        
    def evaluate(
        self,
        test_data: Tuple[np.ndarray, np.ndarray],
        test_metric: Callable = L2,
    ):
        
        '''Compute test data according to some metric'''
        
        (X, y) = test_data
        y_pred = self.compute_forward(X)
        
        return test_metric(y_pred, y) 
        

        
    def prepare_objective_fn(
        self,
        objective_fn: Callable,
    ) -> None:
        
        '''Just to get this ugly shit out of sight'''
        
        # only L2 implemented now
        assert objective_fn == L2
        
        # set objective fn & its derivative as network params. 
        # Not pretty.
        self.objective_fn = objective_fn
        if self.objective_fn == L2:
            self.objective_derivative = L2_deriv
        
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_epochs: int,
        batch_size: Union[float, int],
        learning_rate: float,
        objective_fn: Callable,
        random_seed: int = None,
        test_data: Tuple[np.ndarray, np.ndarray]=None,
        test_metric: Callable = L2,
        verbosity: int=0,
    ) -> None:
        
        # make sure training data is in good shape:
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        assert X_train.shape[0] == y_train.shape[0]
        
        # prepare objective fn and its derivative as network params:
        self.prepare_objective_fn(objective_fn)

        # By setting the random number generator here, the 
        # permutations will be different each time in the loops,
        # but the whole will be reproducible.
        rng = np.random.default_rng(random_seed)
        
        # combine X and y
        Xy_train = np.asarray(list(zip(X_train, y_train))) # UGLYYYYY
        n_data = len(Xy_train)
        # Xy_train = np.column_stack((X_train, y_train))
        # n_data = Xy_train.shape[0]
        if verbosity > 0: print(f'{n_data} training samples')
        
        # outer loop over epochs
        for _i in range(n_epochs):
            
            if verbosity > 0: print(f'Training epoch {_i+1}/{n_epochs}')

            # randomly shuffle data to make sure our batches are different
            shuffled_indices = rng.permutation(n_data)
            Xy_tr = Xy_train[shuffled_indices] # this works for nparray, shuffles first index automatically.
    
            # this works even if batch size doesn't fit into n_data, it
            # just means that the last batch is smaller.
            batches = [Xy_tr[k : batch_size+k] 
                       for k in range(0, n_data, batch_size)]
            if verbosity > 1:
                for _i, batch in batches:
                    print('batch {_i}: \nbatch')
            
            # inner loop over batches within epoch
            for batch in batches:
                self.update_network(batch, learning_rate)
                
            if test_data:
                # print(test_data)
                test_value = self.evaluate(test_data, test_metric=test_metric) 
                # n_correct = test_metric(self, test_data) # TODO why does this only work with self in the call?!
                print(f'Epoch {_i+1}/{n_epochs}: {test_value:.4f}')

    def update_network(
        self,
        data,
        learning_rate
    ) -> None:
        
        n_samples = len(data)
        
        grad_b = []
        grad_w = []
        for layer in self.layers:
            grad_b.append(np.zeros_like(layer.biases))
            grad_w.append(np.zeros_like(layer.weights))
        
        for (x,y) in data:
            # returns a list of gradients to biases and weights,
            # one list element per layer
            sample_grad_b, sample_grad_w = self.compute_sample_gradient(x, y)
            
            # add sample gradients to gradients to biases & weights overall
            # not sure this is the cleverest code.
            grad_b = [gb + sgb for gb, sgb in zip(grad_b, sample_grad_b)]
            grad_w = [gw + sgw for gw, sgw in zip(grad_w, sample_grad_w)]
            
        # finally, update the network using the collected gradient
        # this uses the steepest descent method, nothing fancy.
        for _i, layer in enumerate(self.layers):
            layer.weights -= (learning_rate/n_samples) * grad_w[_i]
            layer.biases -= (learning_rate/n_samples) * grad_b[_i]

    def compute_sample_gradient(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool=False,
    ) -> Tuple[list, list]:
        
        '''
        Compute the gradient for one sample using backpropagation
        through the network.
        '''
        
        if verbose:
            print('working on sample:')
            print(f'X = {X}, y = {y}')
            print(f'X first had shape {X.shape}')
        X = X.reshape(len(X), 1) # this is what the book does. (in mnist_loader)
        # y = y.reshape(1,1) # don't think this is needed...?
        if verbose: print(f'and now has shape {X.shape}, y has shape {y.shape}')
        
        grad_b = []; grad_w = []
        for layer in self.layers:
            grad_b.append(np.zeros_like(layer.biases))
            grad_w.append(np.zeros_like(layer.weights))
        
        # store activations & weighted inputs in lists
        # 'activation' being output of a layer
        a = [X]; z = []
        # forward loop over layers
        for _i, layer in enumerate(self.layers):
            if verbose:
                print(f'forward pass, layer {_i}')
                print(f'  weights shape: {layer.weights.shape} - a prev shape: {a[-1].shape} - b shape: {layer.biases.shape}')
            z.append( np.dot(layer.weights,a[-1])+layer.biases )
            a.append( layer.activation_fn(z[-1]) )
            
        # # intermediate sanity check
        if verbose:
            print('sanity check after forward pass:')
            for act in a: print(f'    (activations shape {act.shape})')
            for zet in z: print(f'    (z (weighted input) shape {zet.shape})')
        
        # backpropagation:
        if verbose: print('backprop, final layer:')
        final_layer = self.layers[-1]
        # delta of final layer:
        delta = self.objective_derivative(a[-1], y) * final_layer.activation_deriv(z[-1])
        if verbose: print(f'  grad_b = delta of shape {delta.shape}')
        grad_b[-1] = delta
        # print(f'...multiplied by a[-2] transpose (was shape {a[-2].shape}, is shape {a[-2].transpose().shape})')
        grad_w[-1] = np.dot(delta, a[-2].transpose())
        if verbose: print(f'  grad_w of shape {grad_w[-1].shape}')
        # compute gradients for remaining layers counting back from final
        for l in range(2, self.n_layers+1):
            if verbose: print(f'backprop, layer {-l} = {self.n_layers-l}')
            # define layers
            this_layer = self.layers[-l]
            next_layer = self.layers[-l+1]
            # compute errors delta, then gradients grad_b and grad_w
            delta = np.dot(next_layer.weights.transpose(), delta) \
                    * this_layer.activation_deriv(z[-l])
            grad_b[-l] = delta
            if verbose: print(f'  grad_b = delta of shape {grad_b[-l].shape})')
            grad_w[-l] = np.dot(delta, a[-l-1].transpose())
            if verbose: print(f'  grad_w of shape {grad_w[-l].shape})')

        return grad_b, grad_w
        
    
    # def define_learning(
    #     self,
    #     objective_fn: Union[str, function] = L2,
    # ) -> None:
        
    #     # only L2 implemented now
    #     assert objective_fn == L2
        
    #     self.objective_fn = objective_fn


