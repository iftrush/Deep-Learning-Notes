# Initialization
### Zero Initialization
 ```python
def initialize_parameters_zeros(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    
    parameters = {}
    L = len(layers_dims) # number of layers in the network
    
    for l in range(1, L):
        parameters['W' + str(l)] = np.zeros((layers_dims[l], layers_dims[l-1]))
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
    return parameters
```
$$
\begin{align}
a&=ReLU(z)=max(0,z)=0\\
\sigma(z)&=\frac{1}{1+e^{-z}}=\frac{1}{1+1}=\frac{1}{2}=y_{pred}\\
\mathcal{L}(a,y)&=-y\ln(y_{pred})-(1-y)\ln(1-y_{pred})\\
\mathcal{L}(0,1)&=-\ln(\frac{1}{2})=0.693\\
\mathcal{L}(0,0)&=-\ln(\frac{1}{2})=0.693
\end{align}
$$
1. fails to break symmetry
### Random Initialization:
```python
def initialize_parameters_random(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    
    np.random.seed(3) # This seed makes sure your "random" numbers will be the as ours
    parameters = {}
    L = len(layers_dims) # integer representing the number of layers
    
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * 10
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters
```

1. Poor initialization can lead to vanishing/exploding gradients
2. Initializing weights to very large random values doesn't work well
-> high weights will cause gradients in sigmoid/ReLU be zero

### Xavier Initialization/He Initialization:
- Xavier Initialization:
$$
W^{[l]}=\sqrt{\frac{1}{n_{l-1}}}
$$
- He Initialization:
$$
W^{[l]}=\sqrt{\frac{2}{n_{l-1}}}
$$
```python
def initialize_parameters_he(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims) - 1 # integer representing the number of layers
     
    for l in range(1, L + 1):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2/layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        
    return parameters
```
1. recommended method