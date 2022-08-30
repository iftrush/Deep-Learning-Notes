# Shallow Neural Network
### Input Layer:
- Single training example $x^{(i)}$:
$$
x^{(i)}=
\begin{bmatrix}
x_1^{(i)}\\
x_2^{(i)}\\
\vdots\\
x_n^{(i)}
\end{bmatrix}\in\mathbb{R}^{n\times 1}
$$
- $m$ training examples: $(x^{(1)}, y^{(1)}),(x^{(2)}, y^{(2)})\dots(x^{(m)}, y^{(m)})$
$$
X=\begin{bmatrix}
\vert & \vert & & \vert \\
x^{(1)} & x^{(2)} & \dots & x^{(m)} \\
\vert & \vert & & \vert
\end{bmatrix}
=\begin{bmatrix}
x_1^{(1)} & x_1^{(2)} & \dots & x_1^{(m)} \\
x_2^{(1)} & x_2^{(2)} & \dots & x_2^{(m)} \\
\vdots & \vdots & \ddots & \vdots \\
x_n^{(1)} & x_n^{(2)} & \dots & x_n^{(m)} \\
\end{bmatrix}\in\mathbb{R}^{n\times m}
$$
### Hidden Layer of Size $n_h$:
- Neuron at layer $l$ for training example $x^{(i)}$:
$$
a^{[l](i)}=
\begin{bmatrix}
a_1^{[l](i)}\\
a_2^{[l](i)}\\
\vdots\\
a_{n_h}^{[l](i)}
\end{bmatrix}\in\mathbb{R}^{n_h\times 1}
$$
- Activation Matrix at layer $l$:
$$
A^{[l]}=
\begin{bmatrix}
\vert & \vert & & \vert \\
a^{[l](1)} & a^{[l](2)} & \dots & a^{[l](m)} \\
\vert & \vert & & \vert
\end{bmatrix}=
\begin{bmatrix}
a_1^{[l](1)} & a_1^{[l](2)} & \dots & a_1^{[l](m)} \\
a_2^{[l](1)} & a_2^{[l](2)} & \dots & a_2^{[l](m)} \\
\vdots & \vdots & \ddots & \vdots\\
a_{n_h}^{[l](1)} & a_{n_h}^{[l](2)} & \dots & a_{n_h}^{[l](m)}
\end{bmatrix}\in\mathbb{R}^{n_h\times m}
$$

### Output Layer:
$$
Y=
\begin{bmatrix}
y^{(1)} & y^{(2)} & \dots & y^{(m)}
\end{bmatrix}=
\begin{bmatrix}
y_1^{(1)} & y_1^{(2)} & \dots & y_1^{(m)} \\
y_2^{(1)} & y_2^{(2)} & \dots & y_2^{(m)} \\
\vdots & \vdots & \ddots & \vdots \\
y_{n_y}^{(1)} & y_{n_y}^{(2)} & \dots & y_{n_y}^{(m)}
\end{bmatrix}
\in\mathbb{R}^{n_y\times m}
$$

### Algorithm:
>Define the NN structure(# of input units, # of hidden units)
>Initialize parameters
>Loop:
>- Implement forward propagation
>- Compute Loss
>- Implement backward propagation to get gradients
>- Update parameters(Gradient Descent)

### Example:
![[classification_kiank.png]]

### Forward Propagation:
- For one example $x^{(i)}$:

$$
\begin{align}
z^{[1](i)}&=
\begin{bmatrix}
z_1^{[1](i)}\\
z_2^{[1](i)}\\
z_3^{[1](i)}\\
z_4^{[1](i)}
\end{bmatrix}\\
&=
\begin{bmatrix}
- & {w_1^{[1]}}^T & -\\
- & {w_2^{[1]}}^T & -\\
- & {w_3^{[1]}}^T & -\\
- & {w_4^{[1]}}^T & -\\
\end{bmatrix}
\begin{bmatrix}
x_1^{(i)}\\
x_2^{(i)}
\end{bmatrix}+
\begin{bmatrix}
b_1^{[1]}\\
b_2^{[1]}\\
b_3^{[1]}\\
b_4^{[1]}
\end{bmatrix}\\
&=W^{[1]}x^{(i)}+b^{[1]}\in\mathbb{R}^{4\times 1}\\

\text{where } &W^{[1]}\in\mathbb{R}^{4\times 2}, x^{(i)}\in\mathbb{R}^{2\times 1}, b\in\mathbb{R}^{4\times 1}\\
a^{[1](i)}&=\tanh(z^{[1](i)})\in\mathbb{R}^{4\times 1}\\
z^{[2](i)}&=W^{[2]}a^{[1](i)}+b^{[2]}\in\mathbb{R}^{1\times 1}\\
\hat{y}&=a^{[2](i)}=\sigma(z^{[2](i)})\in\mathbb{R}^{1\times 1}
\end{align}
$$
- Vectorization:
$$
\begin{align}
Z^{[1]}&=W^{[1]}X+b^{[1]} \in\mathbb{R}^{4\times m}\\
(4\times m)&=(4\times 2)(2\times m)+(4\times 1)\\
A^{[1]}&=\tanh(Z^{[1]})\in\mathbb{R}^{4\times m}\\
Z^{[2]}&=W^{[2]}A^{[1]}+b^{[2]}\in\mathbb{R}^{1\times m}\\
(1\times m)&=(1\times 4)(4\times m)+(1\times 1)\\
\hat{Y}&=A^{[2]}=\sigma(Z^{[2]})\in\mathbb{R}^{1\times m}
\end{align}
$$
$$
\text{Note}:\text{adding }b^{[1]},b^{[2]}\text{ triggers python broadcasting}
$$

### Cost Function:
- Cost:
$$
J=-\frac{1}{m}\sum_{i=1}^{m}
(y^{(i)}\log(a^{[2](i)})+(1-y^{(i)})\log(1-a^{[2](i)}))
$$
- Vectorization:
$$
J=-\frac{1}{m}(Y\log(A^{[2]})^T+(1-Y)\log(1-A^{[2]})^T)
$$

### Activation Functions:
- Sigmoid:
$$
a=\sigma(z)=\frac{1}{1+e^{-z}}
$$
- tanh:
$$
a=\tanh(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}}
$$
- ReLU: *Rectified Linear Unit*
$$
a=\max(0, z)
$$
- Leaky ReLU:
$$
a=\max(0.01z, z)
$$

### Derivatives of Activation Functions:
- Sigmoid:
$$
\begin{align}
a=g(z)&=\frac{1}{1+e^{-z}}\\
g'(z)&=\frac{e^{-z}}{(1+e^{-z})^2}\\
&=(\frac{1}{1+e^{-z}})(\frac{e^{-z}}{1+e^{-z}})\\
&=(\frac{1}{1+e^{-z}})(\frac{1+e^{-z}-1}{1+e^{-z}})\\
&=(\frac{1}{1+e^{-z}})(1-(\frac{1}{1+e^{-z}}))\\
&=g(z)(1-g(z))\\
&=a(1-a)
\end{align}
$$
- tanh:
$$
\begin{align}
a=\tanh(z)&=\frac{e^z-e^{-z}}{e^z+e^{-z}}\\
g'(z)&=\frac{(e^z+e^{-z})(e^z+e^{-z})-(e^z-e^{-z})(e^z-e^{-z})}{(e^z+e^{-z})^2}\\
&=\frac{(e^z+e^{-z})^2-(e^z-e^{-z})^2}{(e^z+e^{-z})^2}\\
&=1-(\frac{e^z-e^{-z}}{e^z+e^{-z}})^2\\
&=1-\tanh^2(z)\\
&=1-a^2
\end{align}
$$
- ReLU:
$$
\begin{align}
g(z)&=\max(0, z)\\
g'(z)&=\begin{cases}
0 & \text{if }z<0\\
1 & \text{if }z>0\\
\text{undef} & \text{if }z=0
\end{cases}
\end{align}
$$
- Leaky ReLU:
$$
\begin{align}
g(z)&=\max(0.01z, z)\\
g'(z)&=\begin{cases}
0.01 & \text{if }z<0\\
1 & \text{if }z\ge0
\end{cases}
\end{align}
$$

### Back Propagation:
- Logistic Regression Gradients:
$$
x,w,b\rightarrow
z=w^Tx+b\rightarrow
a=g=\sigma(z)\rightarrow
\mathcal{L}(a,y)
$$
$$
\begin{align}
da&=\frac{d}{da}\mathcal{L}(a,y)=-y\log a-(1-y)\log(1-a)\\
&=-\frac{y}{a}+\frac{1-y}{1-a}\\
&=\frac{a-y}{a(1-a)}\\
dz&=\frac{\partial\mathcal{L}}{\partial z}\\
&=\frac{\partial\mathcal{L}}{\partial a}\cdot\frac{\partial a}{\partial z}\\
&=da\cdot g'(z)\\
&=(\frac{a-y}{a(1-a)})(a(1-a))\\
&=a-y\quad(\text{only applied to Logistic Regression})\\
db&=dz\\
dw&=\frac{\partial\mathcal{L}}{\partial w}\\
&=\frac{\partial\mathcal{L}}{\partial z}\cdot
\frac{\partial z}{\partial w}\\
&=dz\cdot x
\end{align}
$$
- Neural Network Gradients:
$$
\begin{align}
&x,W^{[1]},b^{[1]}\\
\rightarrow &z^{[1]}=W^{[1]}x+b^{[1]}\\
\rightarrow &a^{[1]}=g^{[1]}(z^{[1]}) \\
\rightarrow &z^{[2]}=W^{[2]}a^{[1]}+b^{[2]}\\
\rightarrow &a^{[2]}=g^{[2]}(z^{[2]})\\
\rightarrow &\mathcal{L}(a^{[2]},y)
\end{align}
$$
$$
\begin{align}
dz^{[2]}&=a^{[2]}-y\in\mathbb{R}^{n^{[2]}}\\
dW^{[2]}&=dz^{[2]}{a^{[1]}}^T\in\mathbb{R}^{n^{[2]}\times n^{[1]}}\\
db^{[2]}&=dz^{[2]}\in\mathbb{R}^{n^{[2]}}\\
dz^{[1]}&=\frac{\partial\mathcal{L}}{\partial a^{[1]}}\cdot\frac{\partial a^{[1]}}{\partial z^{[1]}}\\
&=\frac{\partial\mathcal{L}}{\partial z^{[2]}}\cdot
\frac{\partial z^{[2]}}{\partial a^{[1]}}\circ g^{[1]'}(z^{[1]})\\
&=({W^{[2]}}^Tdz^{[2]})\circ g^{[1]'}(z^{[1]})\in\mathbb{R}^{n^{[1]}\times1}\\
(n^{[1]},1)&=(n^{[1]},n^{[2]})(n^{[2]},1)\circ(n^{[1]},1)\\
dW^{[1]}&=dz^{[1]}x^T\in\mathbb{R}^{n^{[1]}\times n_x}\\
db^{[1]}&=dz^{[1]}\in\mathbb{R}^{n^{[1]}\times1}
\end{align}
$$
- Vectorization:
$$
\begin{align}
dZ^{[2]}&=A^{[2]}-Y\\
dW^{[2]}&=\frac{1}{m}dZ^{[2]}{A^{[1]}}^T\\
db^{[2]}&=\frac{1}{m}np.sum(dZ^{[2]},axis=1,keepdims=True)\\
dZ^{[1]}&=W^{[2]^T}dZ^{[2]}\circ g^{[1]'}(Z^{[1]})\\
dW^{[1]}&=\frac{1}{m}dZ^{[1]}X^T\\
db^{[1]}&=\frac{1}{m}np.sum(dZ^{[1]},axis=1,keepdims=True)\\
\end{align}
$$

### Gradient Descent:
- $\theta=\theta-\alpha\frac{\partial J}{\partial \theta}$:
$$
\begin{align}
W^{[1]}&=W^{[1]}-\alpha\, dW^{[1]}\\
b^{[1]}&=b^{[1]}-\alpha\, db^{[1]}\\
W^{[2]}&=W^{[2]}-\alpha\, dW^{[2]}\\
b^{[2]}&=b^{[2]}-\alpha\, db^{[2]}\\
\end{align}
$$

### Code:
- Initialization:
```python
def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    np.random.seed(2) # we set up a seed so that your output matches ours although the initialization is random.
    

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))


    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters
```
- Forward Propagation:
```python
def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    # Implement Forward Propagation to calculate A2 (probabilities)
    Z1 = W1 @ X + b1
    A1 = np.tanh(Z1)
    Z2 = W2 @ A1 + b2
    A2 = sigmoid(Z2)
    
    assert(A2.shape == (1, X.shape[1]))
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache
```
- Compute Cost:
```python
def compute_cost(A2, Y):
    """
    Computes the cross-entropy cost given in equation (13)
    
    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost given equation (13)
    
    """
    
    m = Y.shape[1] # number of examples

    # Compute the cross-entropy cost
    logprobs = Y @ np.log(A2).T + (1-Y) @ np.log(1-A2).T
    cost = logprobs / (-m)
    
    cost = float(np.squeeze(cost))  # makes sure cost is the dimension we expect. 
                                    # E.g., turns [[17]] into 17 
    
    return cost
```
- Backpropagation:
```python
def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.
    
    Arguments:
    parameters -- python dictionary containing our parameters 
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    
    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]
    
    # First, retrieve W1 and W2 from the dictionary "parameters".
    W1 = parameters['W1']
    W2 = parameters['W2']
        
    # Retrieve also A1 and A2 from dictionary "cache".
    A1 = cache['A1']
    A2 = cache['A2']
    
    # Backward propagation: calculate dW1, db1, dW2, db2. 
    dZ2 = A2 - Y
    dW2 = 1 / m * (dZ2 @ A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T @ dZ2 * (1 - np.power(A1, 2))
    dW1 = 1 / m * (dZ1 @ X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads
```
- Gradient Descent:
```python
def update_parameters(parameters, grads, learning_rate = 1.2):
    """
    Updates parameters using the gradient descent update rule given above
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients 
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    """
    # Retrieve a copy of each parameter from the dictionary "parameters". Use copy.deepcopy(...) for W1 and W2
    W1 = copy.deepcopy(parameters['W1'])
    b1 = copy.deepcopy(parameters['b1'])
    W2 = copy.deepcopy(parameters['W2'])
    b2 = copy.deepcopy(parameters['b2'])
    
    # Retrieve each gradient from the dictionary "grads"
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']
    
    # Update rule for each parameter
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters
```
- Neural Network Model:
```python
def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    
    # Initialize parameters
    parameters = initialize_parameters(n_x, n_h, n_y)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads)
        
        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters
```
- Prediction:
```python
def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    A2, cache = forward_propagation(X, parameters)
    predictions = A2 > 0.5
    
    return predictions
```