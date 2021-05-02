


def addLocation(features, inDim, lDim, outDim = -1, h = None, w = None, locType = "L", mod = "CNCT", name = "", reuse = None): # h,w not needed
    
    with tf.variable_scope("addLocation" + name, reuse = reuse):
        batchSize = tf.shape(features)[0]
        if h is None:
            h = tf.shape(features)[1]
        if w is None:
            w = tf.shape(features)[2]
        dim = inDim

        if mod == "LIN":
            if outDim < 0:
                outDim = dim

            grid, _ = locations[locType](h, w, lDim, outDim = outDim, addBias = False)
            features = linear(features, dim, outDim, name = "LIN")
            features += grid  
            return features, outDim

        if mod == "CNCT":
            grid, lDim = locations[locType](h, w, lDim)
            # grid = tf.zeros_like(features) + grid
            grid = tf.tile(tf.expand_dims(grid, axis = 0), [batchSize, 1, 1, 1])
            features = tf.concat([features, grid], axis = -1)
            dim += lDim

        elif mod == "ADD":
            grid, _ = locations[locType](h, w, lDim, outDim = dim)
            features += grid    
        
        elif mod == "MUL": # MUL
            grid, _ = locations[locType](h, w, lDim, outDim = dim)

            if outDim < 0:
                outDim = dim

            grid = tf.tile(tf.expand_dims(grid, axis = 0), [batchSize, 1, 1, 1])
            features = tf.concat([features, grid, features * grid], axis = -1)
            dim *= 3                

        if outDim > 0:
            features = linear(features, dim, outDim)
            dim = outDim 

    return features, dim