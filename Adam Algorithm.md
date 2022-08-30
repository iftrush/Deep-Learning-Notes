# Adam Algorithm
### Advantages:
- Adam clearly outperforms mini-batch gradient descent and Momentum
- If you run the model for more epochs on this simple dataset, all three methods will lead to very good results. However, you've seen that Adam converges a lot faster
- Relatively low memory requirements (though higher than gradient descent and gradient descent with momentum)
- Usually works well even with little tuning of hyperparameters (except $\alpha$)
### Adaptive Moment Estimation Algorithm:
> $v_{dw}=0, s_{dw}=0, v_{db}=0, s_{db}=0$
> For $l=1,\dots,L$:
> $\qquad$ Compute $dW^{[l]}, db^{[l]}$ on current mini-batch
> $\qquad v_{dW^{[l]}}=\beta_1v_{dW^{[l]}}+(1-\beta_1)dW^{[l]}$
> $\qquad v_{dW^{[l]}}^{corrected}=\frac{v_{dW^{[l]}}}{1-\beta_1^t}$
> $\qquad s_{dW^{[l]}}=\beta_2s_{dW^{[l]}}+(1-\beta_2){dW^{[l]}}^2$
> $\qquad s_{dW^{[l]}}^{corrected}=\frac{s_{dW^{[l]}}}{1-\beta_2^t}$
> $\qquad W^{[l]}=W^{[l]}-\alpha\frac{v_{dW^{[l]}}^{corrected}}{\sqrt{s_{dW^{[l]}}^{corrected}}+\epsilon}$
> $\qquad v_{db^{[l]}}=\beta_1v_{db^{[l]}}+(1-\beta_1)db^{[l]}$
> $\qquad v_{db^{[l]}}^{corrected}=\frac{v_{db^{[l]}}}{1-\beta_1^t}$
> $\qquad s_{db^{[l]}}=\beta_2s_{db^{[l]}}+(1-\beta_2){db^{[l]}}^2$
> $\qquad s_{db^{[l]}}^{corrected}=\frac{s_{db^{[l]}}}{1-\beta_2^t}$
> $\qquad b^{[l]}=b^{[l]}-\alpha\frac{v_{db^{[l]}}^{corrected}}{\sqrt{s_{db^{[l]}}^{corrected}}+\epsilon}$
```python
def initialize_adam(parameters) :
    """
    Initializes v and s as two python dictionaries with:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl
    
    Returns: 
    v -- python dictionary that will contain the exponentially weighted average of the gradient. Initialized with zeros.
                    v["dW" + str(l)] = ...
                    v["db" + str(l)] = ...
    s -- python dictionary that will contain the exponentially weighted average of the squared gradient. Initialized with zeros.
                    s["dW" + str(l)] = ...
                    s["db" + str(l)] = ...

    """
    
    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    s = {}
    
    # Initialize v, s. Input: "parameters". Outputs: "v, s".
    for l in range(1, L + 1):
        v["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
        v["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
        s["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
        s["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
    
    return v, s
```
```python
def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    """
    Update parameters using Adam
    
    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    t -- Adam variable, counts the number of taken steps
    learning_rate -- the learning rate, scalar.
    beta1 -- Exponential decay hyperparameter for the first moment estimates 
    beta2 -- Exponential decay hyperparameter for the second moment estimates 
    epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
    parameters -- python dictionary containing your updated parameters 
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    """
    
    L = len(parameters) // 2                 # number of layers in the neural networks
    v_corrected = {}                         # Initializing first moment estimate, python dictionary
    s_corrected = {}                         # Initializing second moment estimate, python dictionary
    
    # Perform Adam update on all parameters
    for l in range(1, L + 1):
        # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
        v["dW" + str(l)] = beta1 * v["dW" + str(l)] + (1-beta1) * grads['dW' + str(l)]
        v["db" + str(l)] = beta1 * v["db" + str(l)] + (1-beta1) * grads['db' + str(l)]

        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
        v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1-beta1**t)
        v_corrected["db" + str(l)] = v["db" + str(l)] / (1-beta1**t)

        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
        s["dW" + str(l)] = beta2 * s["dW" + str(l)] + (1-beta2) * np.square(grads['dW' + str(l)])
        s["db" + str(l)] = beta2 * s["db" + str(l)] + (1-beta2) * np.square(grads['db' + str(l)])

        # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
        s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1-beta2**t)
        s_corrected["db" + str(l)] = s["db" + str(l)] / (1-beta2**t)

        # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * v_corrected["dW" + str(l)] / (np.sqrt(s_corrected["dW" + str(l)]) + epsilon)
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * v_corrected["db" + str(l)] / (np.sqrt(s_corrected["db" + str(l)]) + epsilon)

    return parameters, v, s, v_corrected, s_corrected
```