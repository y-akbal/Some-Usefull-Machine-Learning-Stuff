class self_attention_with_heads(Layer):
    def __init__(self, heads = 4 ,causal = False, dropout = 0.2, agg = "concat"):
        assert agg in ["concat", "add"], f"agg can be either concat or add, you entered {agg}"
        super().__init__()
        self.agg = agg
        self.heads = [self_attention(causal = causal, dropout = dropout) for _ in range(heads)]
            
    @tf.function()        
    def call(self, inputs, training = None):
        if self.agg == "add":
            x = sum([head(inputs) for head in self.heads])
            return x/len(self.heads)
        else:
            concat = [head(inputs) for head in self.heads]            
            return tf.concat(concat, axis =1)
