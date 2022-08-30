# Neural Style Transfer
- Neural style transfer takes a content image $C$ and a style image $S$ and generates the content image $G$ with the style of style image
- Given a ConvNet like AlexNet
- Pick a unit in layer $1$. Find the nine image patches that maximize the unit's activation
- Repeat for other units and layers.
- In low level layers, more are textures, colors, edges
- In deep layers, entire image like animals will show up
![[dleachlayers.png]]
### Cost Function:

- Content Cost:
	$$
	J_{content}(C,G)=\frac{1}{4\times n_H\times n_W\times n_C}\sum_{\text{all entires}}(a^{(C)}-a^{(G)})^2
	$$
	- Unroll 3D volumns to 2D matrix:
	![[unroll.png]]
```python
def compute_content_cost(content_output, generated_output):
    """
    Computes the content cost
    
    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
    
    Returns: 
    J_content -- scalar that you compute using equation 1 above.
    """
    a_C = content_output[-1]
    a_G = generated_output[-1]
    
    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape a_C and a_G (≈2 lines)
    a_C_unrolled = tf.reshape(a_C, shape=[m, -1, n_C])
    a_G_unrolled = tf.reshape(a_G, shape=[m, -1, n_C])
    
    # compute the cost with tensorflow (≈1 line)
    J_content =  1 / (4 * n_H * n_W * n_C) * tf.reduce_sum(tf.square(a_C_unrolled - a_G_unrolled))
    
    return J_content
```
- Style Cost at layer $l$:
	- Gram Matrix:
	$$
	G_{gram}=A_{unrolled}A_{unrolled}^T
	$$
	- $G_{(gram)i,j}$: correlation
	Measure how "similar" the activations of filter $i$ are to the activations of filter $j$
	- $G_{(gram)i,i}$: prevalence of patterns or textures
	Measure how "active" a filter $i$ is
```python
def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)
    
    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """  
    ### START CODE HERE
    #(≈1 line)
    
    GA = A @ tf.transpose(A)
    
    ### END CODE HERE

    return GA
```
$$
J_{style}^{[l]}(S,G)=\frac{1}{4\times n_C^2\times(n_H\times n_W)^2}\sum_{i=1}^{n_C}\sum_{j=1}^{n_C}(G_{(gram)i,j}^{(S)}-G_{(gram)i,j}^{(G)})^2
$$
```python
def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
    
    Returns: 
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape the images from (n_H * n_W, n_C) to have them of shape (n_C, n_H * n_W) (≈2 lines)
    a_S = tf.transpose(tf.reshape(a_S, shape=[n_H * n_W, n_C]))
    a_G = tf.transpose(tf.reshape(a_G, shape=[n_H * n_W, n_C]))

    # Computing gram_matrices for both images S and G (≈2 lines)
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)
    
    # Computing the loss (≈1 line)
    J_style_layer = 1 / (4 * n_C**2 * n_H**2 * n_W**2) * tf.reduce_sum(tf.square(GS - GG))
    
    return J_style_layer
```
- Total Style Cost:
	- Style Weights: $\lambda^{[l]}$
	
$$
J_{style}(S,G)=\sum_{l}\lambda^{[l]}J_{style}^{[l]}(S,G)
$$
```python
def compute_style_cost(style_image_output, generated_image_output, STYLE_LAYERS=STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers
    
    Arguments:
    style_image_output -- our tensorflow model
    generated_image_output --
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them
    
    Returns: 
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    # initialize the overall style cost
    J_style = 0

    # Set a_S to be the hidden layer activation from the layer we have selected.
    # The last element of the array contains the content layer image, which must not to be used.
    a_S = style_image_output[:-1]

    # Set a_G to be the output of the choosen hidden layers.
    # The last element of the array contains the content layer image, which must not to be used.
    a_G = generated_image_output[:-1]
    for i, weight in zip(range(len(a_S)), STYLE_LAYERS):  
        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S[i], a_G[i])

        # Add weight * J_style_layer of this layer to overall style cost
        J_style += weight[1] * J_style_layer

    return J_style
```
- Total Cost:
$$
J(G)=\alpha J_{content}(C,G)+\beta J_{style}(S,G)
$$
```python
@tf.function()
def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Computes the total cost function
    
    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost
    
    Returns:
    J -- total cost as defined by the formula above.
    """
    J = alpha * J_content + beta * J_style

    return J
```
- Gradient:
```python
optimizer = tf.keras.optimizers.Adam(learning_rate=0.03)

@tf.function()
def train_step(generated_image):
    with tf.GradientTape() as tape:
        # In this function you must use the precomputed encoded images a_S and a_C
        # Compute a_G as the vgg_model_outputs for the current generated image
        a_G = vgg_model_outputs(generated_image)
        
        # Compute the style cost
        J_style = compute_style_cost(a_S, a_G, STYLE_LAYERS=STYLE_LAYERS)

        # Compute the content cost
        J_content = compute_content_cost(a_C, a_G)
        # Compute the total cost
        J = total_cost(J_content, J_style, alpha=10, beta=40)
        
    grad = tape.gradient(J, generated_image)

    optimizer.apply_gradients([(grad, generated_image)])
    generated_image.assign(clip_0_1(generated_image))
    # For grading purposes
    return J
```