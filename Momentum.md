# Momentum
### Algorithm:
> Hyperperameters: $\alpha,\beta=0.9$
>For $l=1,\dots,L$:
>$\qquad$ Compute $dW^{[l]}, db^{[l]}$ on current mini-batch
>$\qquad\,v_{dW^{[l]}}=\beta v_{dW^{[l]}}+(1-\beta)dW^{[l]}$
>$\qquad\,v_{db^{[l]}}=\beta v_{db^{[l]}}+(1-\beta)db^{[l]}$
>$\qquad\,W^{[l]}=W^{[l]}-\alpha v_{dW^{[l]}}$
>$\qquad\,b^{[l]}=b^{[l]}-\alpha v_{db^{[l]}}$

```python
def initialize_velocity(parameters):
    """
    Initializes the velocity as a python dictionary with:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    
    Returns:
    v -- python dictionary containing the current velocity.
                    v['dW' + str(l)] = velocity of dWl
                    v['db' + str(l)] = velocity of dbl
    """
    
    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    
    # Initialize velocity
    for l in range(1, L + 1):
        v["dW" + str(l)] = np.zeros(parameters['W' + str(l)].shape)
        v["db" + str(l)] = np.zeros(parameters['b' + str(l)].shape)
        
    return v
```

```python
def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """
    Update parameters using Momentum
    
    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- python dictionary containing the current velocity:
                    v['dW' + str(l)] = ...
                    v['db' + str(l)] = ...
    beta -- the momentum hyperparameter, scalar
    learning_rate -- the learning rate, scalar
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    v -- python dictionary containing your updated velocities
    """

    L = len(parameters) // 2 # number of layers in the neural networks
    
    # Momentum update for each parameter
    for l in range(1, L + 1):
        v["dW" + str(l)] = beta * v["dW" + str(l)] + (1-beta) * grads['dW' + str(l)]
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * v["dW" + str(l)]
        v["db" + str(l)] = beta * v["db" + str(l)] + (1-beta) * grads['db' + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * v["db" + str(l)]
        
    return parameters, v
```

- The velocity is initialized with zeros. So the algorithm will take a few iterations to "build up" velocity and start to take bigger steps.
- If $\beta=0$, then this just becomes standard gradient descent without momentum
$$
\begin{align}
v_{dW}&=\beta v_{dW}+(1-\beta)dW\\
&=dW\\
W&=W-\alpha v_{dW}\\
&=W-\alpha dW
\end{align}
$$