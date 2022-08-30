# Convolutional Neural Network
### Advantages:
- Parameter Sharing: 
A feature detector that's useful in one part of the image is probably useful in another part of the image.
- Sparsity of Connections: 
In each layer, each output value depends only on a small number of inputs.
### Convolution(CONV):
- Notations:
$$
\begin{align}
n_H&:\text{height}\\
n_W&:\text{weight}\\
n_C&:\#\text{ of channels}\\
f&:\text{filter size}\\
p&:\text{padding}\\
s&:\text{stride}
\end{align}
$$
- Input:
$$
n_H^{[l-1]}\times n_W^{[l-1]}\times n_C^{[l-1]}
$$
- Output:
$$
n_H^{[l]}\times n_W^{[l]}\times n_C^{[l]}
$$
$$
n_H^{[l]}=\Big\lfloor\frac{n_H^{[l-1]}+2p^{[l]}-f^{[l]}}{s^{[l]}}+1\Big\rfloor
$$
$$
n_W^{[l]}=\Big\lfloor\frac{n_W^{[l-1]}+2p^{[l]}-f^{[l]}}{s^{[l]}}+1\Big\rfloor
$$
- Each Filter:
$$
f^{[l]}\times f^{[l]}\times n_c^{[l-1]}
$$
- Activations:
$$
\begin{align}
a^{[l]}&:n_H^{[l]}\times n_W^{[l]}\times n_C^{[l]}\\
A^{[l]}&:m\times n_H^{[l]}\times n_W^{[l]}\times n_C^{[l]}
\end{align}
$$
- Weights:
$$
f^{[l]}\times f^{[l]}\times n_C^{[l-1]}\times n_C^{[l]}
$$
- Bias:
$$
1\times1\times1\times n_C^{[l]}
$$
### Implementation:
- Zero Padding:
```python
def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image, 
    as illustrated in Figure 1.
    
    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions
    
    Returns:
    X_pad -- padded image of shape (m, n_H + 2 * pad, n_W + 2 * pad, n_C)
    """
    X_pad = np.pad(X, ((0,0), (pad,pad), (pad,pad), (0,0)), mode='constant', constant_values=(0,0))
    
    return X_pad
```
- Single Step of Convolution:
```python
def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
    of the previous layer.
    
    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
    
    Returns:
    Z -- a scalar value, the result of convolving the sliding window (W, b) on a slice x of the input data
    """
	# Element-wise product between a_slice_prev and W. Do not add the bias yet.
    s = a_slice_prev * W
	# Sum over all entries of the volume s.
    Z = np.sum(s)
	# Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
    Z += float(b)

    return Z
```
- Convolution Forward:
```python
def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function
    
    Arguments:
    A_prev -- output activations of the previous layer, 
        numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"
        
    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """
    
    # Retrieve dimensions from A_prev's shape (≈1 line)  
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    # Retrieve dimensions from W's shape (≈1 line)
    f, f, n_C_prev, n_C = W.shape
    # Retrieve information from "hparameters" (≈2 lines)
    stride = hparameters['stride']
    pad = hparameters['pad']
    # Compute the dimensions of the CONV output volume using the formula given above. 
    n_H = (n_H_prev - f + 2 * pad) // stride + 1
    n_W = (n_W_prev - f + 2 * pad) // stride + 1
    # Initialize the output volume Z with zeros. (≈1 line)
    Z = np.zeros((m, n_H, n_W, n_C))
    # Create A_prev_pad by padding A_prev
    A_prev_pad = zero_pad(A_prev, pad)
	
	# loop over the batch of training examples
    for i in range(m):
		# Select ith training example's padded activation
        a_prev_pad = A_prev_pad[i]
		# loop over vertical axis of the output volume
        for h in range(n_H):
            vert_start = h * stride
            vert_end = vert_start + f
			# loop over horizontal axis of the output volume
            for w in range(n_W):
                horiz_start = w * stride
                horiz_end = horiz_start + f
				# loop over channels (= #filters) of the output volume
                for c in range(n_C):
					# Use the corners to define the (3D) slice of a_prev_pad
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
					# Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron.
                    weights = W[:, :, :, c]
                    biases = b[:, :, :, c]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, weights, biases)
    
    # Save information in "cache" for the backprop
    cache = (A_prev, W, b, hparameters)
    
    return Z, cache
```
- Convolution Backward
	- Derivation:
	$$
	\begin{align}
	Z_{hw}&=A_{slice}\ast W+b\\
	db&=\sum_{h=0}^{n_H}\sum_{w=0}^{n_W}dZ_{hw}\frac{\partial Z_{hw}}{\partial b}\\
	&=\sum_{h=0}^{n_H}\sum_{w=0}^{n_W}dZ_{hw}\cdot 1\\
	&=\sum_{h=0}^{n_H}\sum_{w=0}^{n_W}dZ_{hw}\\
	dW_{ij}&=\sum_{h=1}^{n_H}\sum_{w=1}^{n_W}dZ_{hw}\frac{\partial Z_{hw}}{\partial W_{ij}}\\
	&=\sum_{h=1}^{n_H}\sum_{w=1}^{n_W}dZ_{hw}\cdot A_{slice,ij}\\
	dW_c&=(\sum_{h=1}^{n_H}\sum_{w=1}^{n_W}dZ_{hw})A_{slice}\\
	&=(dZ_{11}+dZ_{12}+\dots+dZ_{n_Hn_W})A_{slice}\\
	&=A_{slice}dZ_{11}+A_{slice}dZ_{12}+A_{slice}dZ_{n_Hn_W}\\
	&=\sum_{h=1}^{n_H}\sum_{w=1}^{n_W}A_{slice}dZ_{hw}\\
	dA_{ij}&=\sum_{h=1}^{n_H}\sum_{w=1}^{n_W}dZ_{hw}\frac{\partial Z_{hw}}{\partial A_{ij}}\\
	&=\sum_{h=1}^{n_H}\sum_{w=1}^{n_W}dZ_{hw}\cdot W_{ij}\\
	dA&=(\sum_{h=1}^{n_H}\sum_{w=1}^{n_W}dZ_{hw})W_c\\
	&=\sum_{h=1}^{n_H}\sum_{w=1}^{n_W}W_cdZ_{hw}
	\end{align}
	$$



```python
def conv_backward(dZ, cache):
    """
    Implement the backward propagation for a convolution function
    
    Arguments:
    dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward(), output of conv_forward()
    
    Returns:
    dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
               numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    dW -- gradient of the cost with respect to the weights of the conv layer (W)
          numpy array of shape (f, f, n_C_prev, n_C)
    db -- gradient of the cost with respect to the biases of the conv layer (b)
          numpy array of shape (1, 1, 1, n_C)
    """    
	# Retrieve information from "cache"
    A_prev, W, b, hparameters = cache
	# Retrieve dimensions from A_prev's shape
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
	# Retrieve dimensions from W's shape
    f, f, n_C_prev, n_C = W.shape
    # Retrieve information from "hparameters"
    stride = hparameters['stride']
    pad = hparameters['pad']
    # Retrieve dimensions from dZ's shape
    m, n_H, n_W, n_C = dZ.shape
    # Initialize dA_prev, dW, db with the correct shapes
    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)
    # Pad A_prev and dA_prev
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                    dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                    db[:,:,:,c] += dZ[i, h, w, c]
        dA_prev[i, :, :, :] = dA_prev_pad[i, pad:-pad, pad:-pad, :]
    
    # Making sure your output shape is correct
    assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
    
    return dA_prev, dW, db
```


### Pooling(POOL):
$$
\begin{align}
n_H^{[l]}&=\Big\lfloor\frac{n_H^{[l-1]}-f^{[l]}}{s^{[l]}}+1\Big\rfloor\\
n_W^{[l]}&=\Big\lfloor\frac{n_W^{[l-1]}-f^{[l]}}{s^{[l]}}+1\Big\rfloor\\
n_C^{[l]}&=n_C^{[l-1]}\\
\text{mode}&: \text{max or average}
\end{align}
$$
### Implementation:
- Pooling Forward:
```python
def pool_forward(A_prev, hparameters, mode = "max"):
    """
    Implements the forward pass of the pooling layer
    
    Arguments:
    A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    hparameters -- python dictionary containing "f" and "stride"
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
    A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters 
    """
    
    # Retrieve dimensions from the input shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieve hyperparameters from "hparameters"
    f = hparameters["f"]
    stride = hparameters["stride"]
    
    # Define the dimensions of the output
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev
    
    # Initialize output matrix A
    A = np.zeros((m, n_H, n_W, n_C))              
    
	# loop over the training examples
    for i in range(m):
		# loop on the vertical axis of the output volume
        for h in range(n_H):
            vert_start = h * stride
            vert_end = vert_start + f
			# loop on the horizontal axis of the output volume
            for w in range(n_W):
                horiz_start = w * stride
                horiz_end = horiz_start + f
				# loop over the channels of the output volume
                for c in range(n_C):
                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)
    
    # Store the input and hparameters in "cache" for pool_backward()
    cache = (A_prev, hparameters)
    
    # Making sure your output shape is correct
    #assert(A.shape == (m, n_H, n_W, n_C))
    
    return A, cache
```
- Pooling Backward:
	1. Derivation of max pooling:
	$$
	\begin{align}
	A_{hw}&=\max(A_{slice})=A_{slice}\ast M(A_{slice})\\
	dA_{slice,ij}&=\sum_{h=1}^{n_H}\sum_{w=1}^{n_W}dA_{hw}\frac{\partial A_{hw}}{\partial A_{slice,ij}}\\
	&=\sum_{h=1}^{n_H}\sum_{w=1}^{n_W}dA_{hw}\cdot M(A_{slice})_{ij}\\
	dA_{slice}&=\sum_{h=1}^{n_H}\sum_{w=1}^{n_W}M(A_{slice})dA_{hw}
	\end{align}
	$$
	2. Derivation of average pooling:
	$$
	\begin{align}
	dA_{slice,ij}&=\text{np.ones(shape)}\ast \frac{dA_{slice,ij}}{n_Hn_W}\\
	&=\text{distribute}\textunderscore\text{value}(dA_{slice,ij}, \text{shape})\\
	dA_{slice}&=\sum_{h=1}^{n_H}\sum_{w=1}^{n_W}dA_{slice,hw}
	\end{align}
	$$
```python
def create_mask_from_window(x):
    """
    Creates a mask from an input matrix x, to identify the max entry of x.
    
    Arguments:
    x -- Array of shape (f, f)
    
    Returns:
    mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
    """    
    mask = x == np.max(x)
    
    return mask
```
```python
def distribute_value(dz, shape):
    """
    Distributes the input value in the matrix of dimension shape
    
    Arguments:
    dz -- input scalar
    shape -- the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz
    
    Returns:
    a -- Array of size (n_H, n_W) for which we distributed the value of dz
    """    
    # Retrieve dimensions from shape (≈1 line)
    n_H, n_W = shape
    # Compute the value to distribute on the matrix (≈1 line)
    average = dz / (n_H * n_W)
    # Create a matrix where every entry is the "average" value (≈1 line)
    a = np.ones(shape) * average
    
    return a
```
```python
def pool_backward(dA, cache, mode = "max"):
    """
    Implements the backward pass of the pooling layer
    
    Arguments:
    dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
    cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters 
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
    dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
    """
    # Retrieve information from cache (≈1 line)
    A_prev, hparameters = cache
    # Retrieve hyperparameters from "hparameters" (≈2 lines)
    stride = hparameters['stride']
    f = hparameters['f']
    # Retrieve dimensions from A_prev's shape and dA's shape (≈2 lines)
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape
    # Initialize dA_prev with zeros (≈1 line)
    dA_prev = np.zeros(A_prev.shape)

	# loop over the training examples
    for i in range(m):
		# select training example from A_prev
        a_prev = A_prev[i]
		# loop on the vertical axis
        for h in range(n_H):
			# loop on the horizontal axis
            for w in range(n_W):
				# loop over the channels (depth)
                for c in range(n_C):
					# Find the corners of the current "slice" (≈4 lines)
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    if mode == "max":
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        mask = create_mask_from_window(a_prev_slice)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += mask * dA[i, h, w, c]
                    elif mode == "average":
                        da = dA[i, h, w ,c]
                        shape = (f, f)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += distribute_value(da, shape)
    
    # Making sure your output shape is correct
    assert(dA_prev.shape == A_prev.shape)
    
    return dA_prev
```

### TensorFlow Implementation:
##### Sequential API:
- Model:
```python
def happyModel():
    """
    Implements the forward propagation for the binary classification model:
    ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    
    Note that for simplicity and grading purposes, you'll hard-code all the values
    such as the stride and kernel (filter) sizes. 
    Normally, functions should take these values as function parameters.
    
    Arguments:
    None

    Returns:
    model -- TF Keras model (object containing the information for the entire training process) 
    """
    model = tf.keras.Sequential([
			## ZeroPadding2D with padding 3, input shape of 64 x 64 x 3
            tfl.ZeroPadding2D(padding=3, input_shape=(64, 64, 3)),
			## Conv2D with 32 7x7 filters and stride of 1
            tfl.Conv2D(filters=32, kernel_size=7, strides=1),
            ## BatchNormalization for axis 3
			tfl.BatchNormalization(axis=3),
			## ReLU
            tfl.ReLU(),
			## Max Pooling 2D with default parameters
            tfl.MaxPool2D(),
			## Flatten layer
            tfl.Flatten(),
			## Dense layer with 1 unit for output & 'sigmoid' activation
            tfl.Dense(units=1, activation='sigmoid'),
        ])
    
    return model
```
- Compile: optimizer, loss, metrics
```python
happy_model.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])
```
- Summary:
```python
happy_model.summary()
```
- Train(fit):
```python
happy_model.fit(X_train, Y_train, epochs=10, batch_size=16)
```
- Test(evaluate):
```python
happy_model.evaluate(X_test, Y_test)
```
##### Functional API:
- Model:
```python
def convolutional_model(input_shape):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    
    Note that for simplicity and grading purposes, you'll hard-code some values
    such as the stride and kernel (filter) sizes. 
    Normally, functions should take these values as function parameters.
    
    Arguments:
    input_img -- input dataset, of shape (input_shape)

    Returns:
    model -- TF Keras model (object containing the information for the entire training process) 
    """

    input_img = tf.keras.Input(shape=input_shape)

	## CONV2D: 8 filters 4x4, stride of 1, padding 'SAME'
    Z1 = tfl.Conv2D(filters=8, kernel_size=4, strides=1, padding='same')(input_img)
	## RELU
    A1 = tfl.ReLU()(Z1)
	## MAXPOOL: window 8x8, stride 8, padding 'SAME'
    P1 = tfl.MaxPool2D(pool_size=(8, 8), strides=8, padding='same')(A1)
	## CONV2D: 16 filters 2x2, stride 1, padding 'SAME'
    Z2 = tfl.Conv2D(filters=16, kernel_size=2, strides=1, padding='same')(P1)
	## RELU
    A2 = tfl.ReLU()(Z2)
	## MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tfl.MaxPool2D(pool_size=(4, 4), strides=4, padding='same')(A2)
	## FLATTEN
    F = tfl.Flatten()(P2)
	## Dense layer
    ## 6 neurons in output layer. Hint: one of the arguments should be "activation='softmax'" 
    outputs = tfl.Dense(units=6, activation='softmax')(F)

    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model
```
- Complie:
```python
conv_model = convolutional_model((64, 64, 3))
conv_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
```
- Train:
```python
history = conv_model.fit(train_dataset, epochs=100, validation_data=test_dataset)
```
- Plot Loss:
```python
df_loss_acc = pd.DataFrame(history.history)
df_loss= df_loss_acc[['loss','val_loss']]
df_loss.rename(columns={'loss':'train','val_loss':'validation'},inplace=True)
df_loss.plot(title='Model loss',figsize=(12,8)).set(xlabel='Epoch',ylabel='Loss')
```
- Plot Accuracy:
```python
df_acc= df_loss_acc[['accuracy','val_accuracy']]
df_acc.rename(columns={'accuracy':'train','val_accuracy':'validation'},inplace=True)
df_acc.plot(title='Model Accuracy',figsize=(12,8)).set(xlabel='Epoch',ylabel='Accuracy')
```