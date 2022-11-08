
class self_attention_heads(Layer):
    def __init__(self, heads = 5, causal = True, dropout = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.causal = causal 
        self.heads = heads
        self.dropout = Dropout(dropout)
        self.conv2d = Conv2D(1, kernel_size = 1, strides = 1, use_bias= True, kernel_initializer= tf.keras.initializers.Constant(
    value=1/self.heads))
        
    def build(self, input_shape, **kwargs):
       
        input_shape = input_shape[0]
        shape = self.heads, input_shape[-2], input_shape[-2]
        initializer = tf.keras.initializers.Orthogonal() #### we in particular ortogonal initialization, in the case that it is needed it will train it!
        initial_value = initializer(shape = shape)
        
        self.kernel = tf.Variable(initial_value = initial_value, trainable = True)
        
        if self.causal: ### this part is used to kill attention of future to past, 
            minf = -tf.constant(20000.0)  ### take this dude to kill softmax maybe a little bit smaller.
            mask = tf.fill(shape, minf)
            self.upper_m = minf - tf.linalg.band_part(mask, num_lower = -1, num_upper = 0)
            
    @tf.function       
    def call(self, inputs, training = None):
        if training:
            inputs = self.dropout(inputs, training) ### dropout is applied in the begining of the layer
            
           
            
        inputs_ = [0 for i in range(len(inputs))]
        
        
        inputs_[0] = tf.expand_dims(inputs[0], axis = -3)
        inputs_[1] = tf.expand_dims(inputs[1], axis = -3)
        
        sim1 = self.kernel @ inputs_[0]
        sim2 = self.kernel @ inputs_[1]
        
                
        att_scores = tf.matmul(sim1, sim2, transpose_b = True)
        
        if self.causal:
            att_scores += self.upper_m
        
        softmaxed = tf.nn.softmax(att_scores, axis = -2)
        
        similarity_heads = softmaxed @ inputs_[1]
        
        transposed = tf.transpose(similarity_heads, [0, 2, 3, 1])
        return tf.squeeze(self.conv2d(transposed), -1)
        
