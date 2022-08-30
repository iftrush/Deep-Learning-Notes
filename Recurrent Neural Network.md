# Recurrent Neural Network
### Sequence Data:
- Speech Recognition
- Music Generation(one-to-many)
- Sentiment Classification(many-to-one)
- DNA Sequence Analysis
- Machine Translation(many-to-many)
- Video Activity Recognition
- Name Entity Recognition

### Notation:
- Superscript $[l]$: $l$th layer
- Superscript $(i)$: $i$th example
- Superscript $\langle t\rangle$: $t$th time step
- Subscript $i$: $i$th entry of a vector

### RNN Cell Forward:
![[rnn_cell.png]]
$$
\begin{align}
a^{\langle t\rangle}&=\tanh(W_{ax}x^{\langle t\rangle}+W_{aa}a^{\langle t-1\rangle}+b_a)\\
\hat{y}^{\langle t\rangle}&=softmax(W_{ya}a^{\langle t\rangle}+b_y)
\end{align}
$$

```python
def rnn_cell_forward(xt, a_prev, parameters):
    """
    Implements a single forward step of the RNN-cell as described in Figure (2)

    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        ba --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, a_prev, xt, parameters)
    """
    
    # Retrieve parameters from "parameters"
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]
    
    # compute next activation state using the formula given above
    a_next = np.tanh(Wax @ xt + Waa @ a_prev + ba)
    # compute output of the current cell using the formula given above
    yt_pred = softmax(Wya @ a_next + by)
    
    # store values you need for backward propagation in cache
    cache = (a_next, a_prev, xt, parameters)
    
    return a_next, yt_pred, cache
```

### RNN Forward Propagation:
![[rnn_forward.png]]
```python
def rnn_forward(x, a0, parameters):
    """
    Implement the forward propagation of the recurrent neural network described in Figure (3).

    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
    parameters -- python dictionary containing:
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        ba --  Bias numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
    a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
    y_pred -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of caches, x)
    """
    
    # Initialize "caches" which will contain the list of all caches
    caches = []
    
    # Retrieve dimensions from shapes of x and parameters["Wya"]
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape
    
    # initialize "a" and "y_pred" with zeros (≈2 lines)
    a = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))
    
    # Initialize a_next (≈1 line)
    a_next = a0
    
    # loop over all time-steps
    for t in range(T_x):
        # Update next hidden state, compute the prediction, get the cache (≈1 line)
        a_next, yt_pred, cache = rnn_cell_forward(x[..., t], a_next, parameters)
        # Save the value of the new "next" hidden state in a (≈1 line)
        a[:,:,t] = a_next
        # Save the value of the prediction in y (≈1 line)
        y_pred[:,:,t] = yt_pred
        # Append "cache" to "caches" (≈1 line)
        caches.append(cache)
    
    # store values needed for backward propagation in cache
    caches = (caches, x)
    
    return a, y_pred, caches
```
### RNN Cell Backward:
![[rnn_cell_backward.png]]
![[rnn_backward.png]]
$$
\begin{align}
a^{\langle t\rangle}&=\tanh(W_{ax}x^{\langle t\rangle}+W_{aa}a^{\langle t-1\rangle}+b_a)\\
\frac{\partial\tanh(x)}{\partial x}&=1-\tanh^2(x)\\
d\tanh&=da_{nezt}\,\circ(1-\tanh^2(W_{ax}x^{\langle t\rangle}+W_{aa}a^{\langle t-1\rangle}+b_a))\\
dW_{ax}&=d\tanh\cdot {x^{\langle t\rangle}}^T\\
dW_{aa}&=d\tanh\cdot {a^{\langle t-1\rangle}}^T\\
db_a&=\sum_{batch}d\tanh\\
dx^{\langle t\rangle}&=W_{ax}^T\cdot d\tanh\\
da_{prev}&=W_{aa}^T\cdot d\tanh\\
\end{align}
$$
```python
def rnn_cell_backward(da_next, cache):
    """
    Implements the backward pass for the RNN-cell (single time-step).

    Arguments:
    da_next -- Gradient of loss with respect to next hidden state
    cache -- python dictionary containing useful values (output of rnn_cell_forward())

    Returns:
    gradients -- python dictionary containing:
                        dx -- Gradients of input data, of shape (n_x, m)
                        da_prev -- Gradients of previous hidden state, of shape (n_a, m)
                        dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                        dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                        dba -- Gradients of bias vector, of shape (n_a, 1)
    """
    
    # Retrieve values from cache
    (a_next, a_prev, xt, parameters) = cache
    
    # Retrieve values from parameters
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    # compute the gradient of dtanh term using a_next and da_next (≈1 line)
    dtanh = da_next * (1 - np.tanh(Wax @ xt + Waa @ a_prev + ba) ** 2)

    # compute the gradient of the loss with respect to Wax (≈2 lines)
    dxt = Wax.T @ dtanh
    dWax = dtanh @ xt.T

    # compute the gradient with respect to Waa (≈2 lines)
    da_prev = Waa.T @ dtanh
    dWaa = dtanh @ a_prev.T

    # compute the gradient with respect to b (≈1 line)
    dba = np.sum(dtanh, axis=1, keepdims=True)
    
    # Store the gradients in a python dictionary
    gradients = {"dxt": dxt, "da_prev": da_prev, "dWax": dWax, "dWaa": dWaa, "dba": dba}
    
    return gradients
```
### RNN Backward Propagation:
```python
def rnn_backward(da, caches):
    """
    Implement the backward pass for a RNN over an entire sequence of input data.

    Arguments:
    da -- Upstream gradients of all hidden states, of shape (n_a, m, T_x)
    caches -- tuple containing information from the forward pass (rnn_forward)
    
    Returns:
    gradients -- python dictionary containing:
                        dx -- Gradient w.r.t. the input data, numpy-array of shape (n_x, m, T_x)
                        da0 -- Gradient w.r.t the initial hidden state, numpy-array of shape (n_a, m)
                        dWax -- Gradient w.r.t the input's weight matrix, numpy-array of shape (n_a, n_x)
                        dWaa -- Gradient w.r.t the hidden state's weight matrix, numpy-arrayof shape (n_a, n_a)
                        dba -- Gradient w.r.t the bias, of shape (n_a, 1)
    """
    # Retrieve values from the first cache (t=1) of caches (≈2 lines)
    (caches, x) = caches
    (a1, a0, x1, parameters) = caches[0]
    
    # Retrieve dimensions from da's and x1's shapes (≈2 lines)
    n_a, m, T_x = da.shape
    n_x, m = x1.shape 
    
    # initialize the gradients with the right sizes (≈6 lines)
    dx = np.zeros((n_x, m, T_x))
    dWax = np.zeros((n_a, n_x))
    dWaa = np.zeros((n_a, n_a))
    dba = np.zeros((n_a, 1))
    da0 = np.zeros((n_a, m))
    da_prevt = np.zeros((n_a, m))
    
    # Loop through all the time steps
    for t in reversed(range(T_x)):
        # Compute gradients at time step t. Choose wisely the "da_next" and the "cache" to use in the backward propagation step. (≈1 line)
        gradients = rnn_cell_backward(da[..., t] + da_prevt, caches[t])
        # Retrieve derivatives from gradients (≈ 1 line)
        dxt, da_prevt, dWaxt, dWaat, dbat = gradients["dxt"], gradients["da_prev"], gradients["dWax"], gradients["dWaa"], gradients["dba"]
        # Increment global derivatives w.r.t parameters by adding their derivative at time-step t (≈4 lines)
        dx[:, :, t] = dxt
        dWax += dWaxt  
        dWaa += dWaat 
        dba += dbat  
        
    # Set da0 to the gradient of a which has been backpropagated through all time-steps (≈1 line) 
    da0 = da_prevt

    # Store the gradients in a python dictionary
    gradients = {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa,"dba": dba}
    
    return gradients
```
### Long Short-Term Memory(LSTM):
![[LSTM.png]]
##### Forget Gate $\Gamma_f$: $f_t$
$$
\Gamma_f^{\langle t\rangle}=\sigma(W_f[a^{\langle t-1\rangle},x^{\langle t\rangle}]+b_f)
$$
- If the subject changes its state (from a singular word to a plural word), the memory of the previous state becomes outdated, so you'll "forget" that outdated state.
- The "forget gate" is a tensor containing values between 0 and 1.
	- If a unit in the forget gate has a value close to 0, the LSTM will "forget" the stored state in the corresponding unit of the previous cell state.
	- If a unit in the forget gate has a value close to 1, the LSTM will mostly remember the corresponding value in the stored state.
##### Candidate Value $\tilde{c}^{\langle t\rangle}$: $\tilde{c}_t$
$$
\tilde{c}^{\langle t\rangle}=\tanh(W_c[a^{\langle t-1\rangle},x^{\langle t\rangle}]+b_c)
$$
- The candidate value is a tensor containing information from the current time step that **may** be stored in the current cell state $c^{\langle t\rangle}$
- The parts of the candidate value that get passed on depend on the update gate
- The candidate value is a tensor containing values that range from -1 to 1
##### Update Gate $\Gamma_i$: $i_t$
$$
\Gamma_i^{\langle t\rangle}=\sigma(W_i[a^{\langle t-1\rangle},x^{\langle t\rangle}]+b_i)
$$
- You use the update gate to decide what aspects of the candidate $\tilde{c}^{\langle t\rangle}$ to add to the cell state $c^{\langle t\rangle}$
- The update gate is a tensor containing values between 0 and 1
	- When a unit in the update gate is close to 1, it allows the value of the candidate $\tilde{c}^{\langle t\rangle}$ to be passed onto the hidden state $c^{\langle t\rangle}$
	- When a unit in the update gate is close to 0, it prevents the corresponding value in the candidate from being passed onto the hidden state
##### Cell State $c^{\langle t\rangle}$: $c_t$
$$
c^{\langle t\rangle}=\Gamma_f^{\langle t\rangle}\circ c^{\langle t-1\rangle}+\Gamma_i^{\langle t\rangle}\circ \tilde{c}^{\langle t\rangle}
$$
- The cell state is the "memory" that gets passed onto future time steps
- The new cell state $c^{\langle t\rangle}$ is a combination of the previous cell state and the candidate value
- The previous cell state $c^{\langle t-1\rangle}$ is adjusted (weighted) by the forget gate $\Gamma_f^{\langle t\rangle}$
- The candidate value $\tilde{c}^{\langle t\rangle}$, adjusted (weighted) by the update gate $\Gamma_i^{\langle t\rangle}$
##### Output Gate $\Gamma_o$: $o_t$
$$
\Gamma_o^{\langle t\rangle}=\sigma(W_o[a^{\langle t-1\rangle},x^{\langle t\rangle}]+b_o)
$$
- The output gate decides what gets sent as the prediction (output) of the time step.
- The output gate is like the other gates, in that it contains values that range from 0 to 1
##### Hidden State $a^{\langle t\rangle}$: $h_t$
$$
a^{\langle t\rangle}=\Gamma_o^{\langle t\rangle}\circ\tanh(c^{\langle t\rangle})
$$
- The hidden state gets passed to the LSTM cell's next time step
- It is used to determine the three gates $(\Gamma_f,\Gamma_i,\Gamma_o)$ of the next time step
- The hidden state is also used for the prediction $y^{\langle t\rangle}$
- The output gate acts like a "mask" that either preserves the values of $\tanh(c^{\langle t\rangle})$ or keeps those values from being included in the hidden state
##### Prediction $y_{pred}^{\langle t\rangle}$:
$$
y_{pred}^{\langle t\rangle}=softmax(W_ya^{\langle t\rangle}+b_y)
$$
### LSTM Cell Forward:
```python
def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    """
    Implement a single forward step of the LSTM-cell as described in Figure (4)

    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    c_prev -- Memory state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc --  Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo --  Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
                        
    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    c_next -- next memory state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, c_next, a_prev, c_prev, xt, parameters)
    
    Note: ft/it/ot stand for the forget/update/output gates, cct stands for the candidate value (c tilde),
          c stands for the cell state (memory)
    """

    # Retrieve parameters from "parameters"
    Wf = parameters["Wf"] # forget gate weight
    bf = parameters["bf"]
    Wi = parameters["Wi"] # update gate weight (notice the variable name)
    bi = parameters["bi"] # (notice the variable name)
    Wc = parameters["Wc"] # candidate value weight
    bc = parameters["bc"]
    Wo = parameters["Wo"] # output gate weight
    bo = parameters["bo"]
    Wy = parameters["Wy"] # prediction weight
    by = parameters["by"]
    
    # Retrieve dimensions from shapes of xt and Wy
    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    # Concatenate a_prev and xt (≈1 line)
    concat = np.vstack((a_prev, xt))

    # Compute values for ft, it, cct, c_next, ot, a_next using the formulas given figure (4) (≈6 lines)
    ft = sigmoid(Wf @ concat + bf)
    it = sigmoid(Wi @ concat + bi)
    cct = np.tanh(Wc @ concat + bc)
    c_next = ft * c_prev + it * cct
    ot = sigmoid(Wo @ concat + bo)
    a_next = ot * np.tanh(c_next)
    
    # Compute prediction of the LSTM cell (≈1 line)
    yt_pred = softmax(Wy @ a_next + by)

    # store values needed for backward propagation in cache
    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

    return a_next, c_next, yt_pred, cache
```
### LSTM Forward Propagation:
![[LSTM_forward.png]]
```python
def lstm_forward(x, a0, parameters):
    """
    Implement the forward propagation of the recurrent neural network using an LSTM-cell described in Figure (4).

    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc -- Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo -- Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
                        
    Returns:
    a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
    y -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
    c -- The value of the cell state, numpy array of shape (n_a, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of all the caches, x)
    """

    # Initialize "caches", which will track the list of all the caches
    caches = []
    Wy = parameters['Wy'] # saving parameters['Wy'] in a local variable in case students use Wy instead of parameters['Wy']
    # Retrieve dimensions from shapes of x and parameters['Wy'] (≈2 lines)
    n_x, m, T_x = x.shape
    n_y, n_a = Wy.shape
    
    # initialize "a", "c" and "y" with zeros (≈3 lines)
    a = np.zeros((n_a, m, T_x))
    c = np.zeros((n_a, m, T_x))
    y = np.zeros((n_y, m, T_x))
    
    # Initialize a_next and c_next (≈2 lines)
    a_next = a0
    c_next = np.zeros((n_a, m))
    
    # loop over all time-steps
    for t in range(T_x):
        # Get the 2D slice 'xt' from the 3D input 'x' at time step 't'
        xt = x[:,:,t]
        # Update next hidden state, next memory state, compute the prediction, get the cache (≈1 line)
        a_next, c_next, yt, cache = lstm_cell_forward(xt, a_next, c_next, parameters)
        # Save the value of the new "next" hidden state in a (≈1 line)
        a[:,:,t] = a_next
        # Save the value of the next cell state (≈1 line)
        c[:,:,t]  = c_next
        # Save the value of the prediction in y (≈1 line)
        y[:,:,t] = yt
        # Append the cache into caches (≈1 line)
        caches.append(cache)
    
    # store values needed for backward propagation in cache
    caches = (caches, x)

    return a, y, c, caches
```
### LSTM Cell Backward:
![[LSTM_cell_backward.png]]
$$
\begin{align}
a^{\langle t\rangle}&=\Gamma_o^{\langle t\rangle}\odot\tanh(c^{\langle t\rangle})\\
c^{\langle t\rangle}&=\Gamma_f^{\langle t\rangle}\odot c^{\langle t-1\rangle}+\Gamma_i^{\langle t\rangle}\odot \tilde{c}^{\langle t\rangle}\\
\tilde{c}^{\langle t\rangle}&=\tanh(W_c[a^{\langle t-1\rangle},x^{\langle t\rangle}]+b_c)\\
\Gamma_o^{\langle t\rangle}&=\sigma(W_o[a^{\langle t-1\rangle},x^{\langle t\rangle}]+b_o)\\
\Gamma_i^{\langle t\rangle}&=\sigma(W_i[a^{\langle t-1\rangle},x^{\langle t\rangle}]+b_i)\\
\Gamma_f^{\langle t\rangle}&=\sigma(W_f[a^{\langle t-1\rangle},x^{\langle t\rangle}]+b_f)\\
d\gamma_o^{\langle t\rangle}&=da_{next}\frac{\partial a^{\langle t\rangle}}{\partial \gamma_o^{\langle t\rangle}}\\
&=da_{next}\odot\tanh(c_{next})\odot\Gamma_o^{\langle t\rangle}\odot(1-\Gamma_o^{\langle t\rangle})\\
\frac{\partial L}{\partial c^{\langle t\rangle}}&=\frac{\partial L'}{\partial c^{\langle t\rangle}}+\frac{\partial L}{\partial a^{\langle t\rangle}}\frac{\partial a^{\langle t\rangle}}{\partial c^{\langle t\rangle}}\\
&=dc_{next}+da_{next}\odot\Gamma_o^{\langle t\rangle}\odot(1-\tanh^2(c_{next}))\\
d\tilde{c}^{\langle t\rangle}&=\frac{\partial L}{\partial c^{\langle t\rangle}}\frac{\partial c^{\langle t\rangle}}{\partial \tilde{c}^{\langle t\rangle}}\\
&=\left(dc_{next}+da_{next}\odot\Gamma_o^{\langle t\rangle}\odot(1-\tanh^2(c_{next}))\right)\odot\left(\Gamma_i^{\langle t\rangle}\odot(1-(\tilde{c}^{\langle t\rangle})^2)\right)\\
&=\left(dc_{next}\odot\Gamma_i^{\langle t\rangle}+\Gamma_o^{\langle t\rangle}\odot(1-\tanh^2(c_{next}))\odot\Gamma_i^{\langle t\rangle}\odot da_{next}\right)\odot\left(1-(\tilde{c}^{\langle t\rangle})^2\right)\\
\frac{\partial L}{\partial \gamma_i^{\langle t\rangle}}&=
\frac{\partial L}{\partial c^{\langle t\rangle}}
\frac{\partial c^{\langle t\rangle}}{\partial\gamma_i^{\langle t\rangle}}\\
d\gamma_i^{\langle t\rangle}&=\left(dc_{next}+da_{next}\odot\Gamma_o^{\langle t\rangle}\odot(1-\tanh^2(c_{next}))\right)\odot\left(\Gamma_i^{\langle t\rangle}\odot (1-\Gamma_i^{\langle t\rangle})\right)\odot\tilde{c}^{\langle t\rangle}\\
&=\left(dc_{next}\odot\tilde{c}^{\langle t\rangle}+\Gamma_0^{\langle t\rangle}\odot(1-\tanh^2(c_{next}))\odot\tilde{c}^{\langle t\rangle}\odot da_{next}\right)\odot\Gamma_i^{\langle t\rangle}\odot(1-\Gamma_i^{\langle t\rangle})\\
\frac{\partial L}{\partial \gamma_f^{\langle t\rangle}}&=
\frac{\partial L}{\partial c^{\langle t\rangle}}
\frac{\partial c^{\langle t\rangle}}{\partial \gamma_f^{\langle t\rangle}}\\
d\gamma_f^{\langle t\rangle}&=\left(dc_{next}+da_{next}\odot\Gamma_o^{\langle t\rangle}\odot(1-\tanh^2(c_{next}))\right)\odot\left(\Gamma_f^{\langle t\rangle}\odot(1-\Gamma_f^{\langle t\rangle})\right)\odot c_{prev}\\
&=\left(dc_{next}\odot c_{prev}+\Gamma_0^{\langle t\rangle}\odot(1-\tanh^2(c_{next}))\odot c_{prev}\odot da_{next}\right)\odot\Gamma_f^{\langle t\rangle}\odot(1-\Gamma_f^{\langle t\rangle})\\
dW_f&=d\gamma_f^{\langle t\rangle}\begin{bmatrix}
a_{prev}\\
x_t
\end{bmatrix}^t\\

dW_i&=d\gamma_i^{\langle t\rangle}\begin{bmatrix}
a_{prev}\\
x_t
\end{bmatrix}^t\\
dW_c&=d\tilde{c}^{\langle t\rangle}\begin{bmatrix}
a_{prev}\\
x_t
\end{bmatrix}^t\\
dW_o&=d\gamma_o^{\langle t\rangle}\begin{bmatrix}
a_{prev}\\
x_t
\end{bmatrix}^t\\
db_f&=\sum_{batch}d\gamma_f^{\langle t\rangle}\\
db_i&=\sum_{batch}d\gamma_i^{\langle t\rangle}\\
db_c&=\sum_{batch}d\tilde{c}^{\langle t\rangle}\\
db_o&=\sum_{batch}d\gamma_o^{\langle t\rangle}\\
da_{prev}&=W_f^Td\gamma_f^{\langle t\rangle}
+W_i^Td\gamma_i^{\langle t\rangle}
+W_c^Td\tilde{c}^{\langle t\rangle}
+W_o^Td\gamma_o^{\langle t\rangle}\\
dc_{prev}&=dc_{next}\odot\Gamma_f^{\langle t\rangle}+\Gamma_o^{\langle t\rangle}\odot(1-\tanh^2(c_{next}))\odot\Gamma_f^{\langle t\rangle}\odot da_{next}\\
dx^{\langle t\rangle}&=W_f^Td\gamma_f^{\langle t\rangle}+
W_u^Td\gamma_u^{\langle t\rangle}+
W_c^Td\tilde{c}^{\langle t\rangle}+
W_o^Td\gamma_o^{\langle t\rangle}
\end{align}
$$
```python
def lstm_cell_backward(da_next, dc_next, cache):
    """
    Implement the backward pass for the LSTM-cell (single time-step).

    Arguments:
    da_next -- Gradients of next hidden state, of shape (n_a, m)
    dc_next -- Gradients of next cell state, of shape (n_a, m)
    cache -- cache storing information from the forward pass

    Returns:
    gradients -- python dictionary containing:
                        dxt -- Gradient of input data at time-step t, of shape (n_x, m)
                        da_prev -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
                        dc_prev -- Gradient w.r.t. the previous memory state, of shape (n_a, m, T_x)
                        dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        dWi -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
                        dWo -- Gradient w.r.t. the weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
                        dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
                        dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
                        dbo -- Gradient w.r.t. biases of the output gate, of shape (n_a, 1)
    """

    # Retrieve information from "cache"
    (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = cache
    
    # Retrieve dimensions from xt's and a_next's shape (≈2 lines)
    n_x, m = xt.shape
    n_a, m = a_next.shape
    
    # Compute gates related derivatives. Their values can be found by looking carefully at equations (7) to (10) (≈4 lines)
    dot = da_next * np.tanh(c_next) * ot * (1 - ot)
    dcct = (dc_next * it + ot * (1 - np.tanh(c_next)**2) * it * da_next) * (1 - cct**2)
    dit = (dc_next * cct + ot * (1 - np.tanh(c_next)**2) * cct * da_next) * it * (1 - it)
    dft = (dc_next * c_prev + ot * (1 - np.tanh(c_next)**2) * c_prev * da_next) * ft * (1 - ft)
    
    # Compute parameters related derivatives. Use equations (11)-(18) (≈8 lines)
    dWf = dft @ np.vstack((a_prev, xt)).T
    dWi = dit @ np.vstack((a_prev, xt)).T
    dWc = dcct @ np.vstack((a_prev, xt)).T
    dWo = dot @ np.vstack((a_prev, xt)).T
    dbf = np.sum(dft, axis=1, keepdims=True)
    dbi = np.sum(dit, axis=1, keepdims=True)
    dbc = np.sum(dcct, axis=1, keepdims=True)
    dbo = np.sum(dot, axis=1, keepdims=True)

    # Compute derivatives w.r.t previous hidden state, previous memory state and input. Use equations (19)-(21). (≈3 lines)
    da_prev = parameters['Wf'][:, :n_a].T @ dft + parameters['Wi'][:, :n_a].T @ dit + parameters['Wc'][:, :n_a].T @ dcct + parameters['Wo'][:, :n_a].T @ dot
    dc_prev = dc_next * ft + ot * (1 - np.tanh(c_next)**2) * ft * da_next
    dxt = parameters['Wf'][:, n_a:].T @ dft + parameters['Wi'][:, n_a:].T @ dit + parameters['Wc'][:, n_a:].T @ dcct + parameters['Wo'][:, n_a:].T @ dot    
    
    # Save gradients in dictionary
    gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
                "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo}

    return gradients
```
### LSTM Backward Propagation:
```python
def lstm_backward(da, caches):
    
    """
    Implement the backward pass for the RNN with LSTM-cell (over a whole sequence).

    Arguments:
    da -- Gradients w.r.t the hidden states, numpy-array of shape (n_a, m, T_x)
    caches -- cache storing information from the forward pass (lstm_forward)

    Returns:
    gradients -- python dictionary containing:
                        dx -- Gradient of inputs, of shape (n_x, m, T_x)
                        da0 -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
                        dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        dWi -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
                        dWo -- Gradient w.r.t. the weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
                        dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
                        dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
                        dbo -- Gradient w.r.t. biases of the output gate, of shape (n_a, 1)
    """

    # Retrieve values from the first cache (t=1) of caches.
    (caches, x) = caches
    (a1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters) = caches[0]

    # Retrieve dimensions from da's and x1's shapes (≈2 lines)
    n_a, m, T_x = da.shape
    n_x, m = x1.shape
    
    # initialize the gradients with the right sizes (≈12 lines)
    dx = np.zeros((n_x, m, T_x))
    da0 = np.zeros((n_a, m))
    da_prevt = np.zeros((n_a, m))
    dc_prevt = np.zeros((n_a, m))
    dWf = np.zeros((n_a, n_a + n_x))
    dWi = np.zeros((n_a, n_a + n_x))
    dWc = np.zeros((n_a, n_a + n_x))
    dWo = np.zeros((n_a, n_a + n_x))
    dbf = np.zeros((n_a, 1))
    dbi = np.zeros((n_a, 1))
    dbc = np.zeros((n_a, 1))
    dbo = np.zeros((n_a, 1))
    
    # loop back over the whole sequence
    for t in reversed(range(T_x)):
        # Compute all gradients using lstm_cell_backward
        gradients = lstm_cell_backward(da[..., t] + da_prevt, dc_prevt, caches[t])
        # Store or add the gradient to the parameters' previous step's gradient
        da_prevt = gradients['da_prev']
        dc_prevt = gradients['dc_prev']
        dx[:,:,t] = gradients['dxt']
        dWf += gradients['dWf']
        dWi += gradients['dWi']
        dWc += gradients['dWc']
        dWo += gradients['dWo']
        dbf += gradients['dbf']
        dbi += gradients['dbi']
        dbc += gradients['dbc']
        dbo += gradients['dbo']
    # Set the first activation's gradient to the backpropagated gradient da_prev.
    da0 = gradients['da_prev']

    # Store the gradients in a python dictionary
    gradients = {"dx": dx, "da0": da0, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
                "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo}
    
    return gradients
```