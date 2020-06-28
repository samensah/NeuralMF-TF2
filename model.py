from tensorflow import keras

class NeuralMF(keras.Model):
    def __init__(self, opt):
        super().__init__() # handles standard args (e.g., name)
        self.opt = opt
        self.K = opt['K']
        self.activation = keras.layers.ReLU()

        # create embedding layers
        self.mf_user_embed  = keras.layers.Embedding(opt['user_num']+1, opt['emb_dim'])
        self.mf_item_embed  = keras.layers.Embedding(opt['item_num']+1, opt['emb_dim'])
        self.mlp_user_embed = keras.layers.Embedding(opt['user_num']+1, opt['emb_dim'])
        self.mlp_item_embed = keras.layers.Embedding(opt['item_num']+1, opt['emb_dim'])

        # mlp layers
        self.mlp = []
        for i in range(0, opt['mlp_layers'], 1):
            self.mlp.extend([keras.layers.Dense(self.K, activation="elu", kernel_initializer="he_normal")])
        self.mlp.extend([keras.layers.Dense(1, activation="elu", kernel_initializer="he_normal")])


    def call(self, inputs):
        # Two inputs, two output
        user_id, item_id = inputs[0], inputs[1]
        xmfu = self.mf_user_embed(user_id)
        xmfi = self.mf_item_embed(item_id)
        xmf = xmfu * xmfi

        xmlpu = self.mlp_user_embed(user_id)
        xmlpi = self.mlp_item_embed(item_id)
        xmlp = keras.layers.concatenate([xmlpu, xmlpi])
        for i, layer in enumerate(self.mlp[:-1]):
            xmlp = layer(xmlp)
            xmlp = self.activation(xmlp)

        x = keras.layers.concatenate([xmf, xmlp])
        x = self.mlp[-1](x)
        return x





