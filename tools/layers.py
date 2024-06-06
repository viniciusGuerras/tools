import numpy as np

class Convolutional:
    def __init__(self, num_kernels, depth, kernel_width=3, kernel_height=3, padding=0, stride=1):
        self.num_kernels = num_kernels 
        self.kernel_width = kernel_width
        self.kernel_height = kernel_height 
        self.padding = padding 
        self.stride = stride
        self.kernels = np.random.randn(num_kernels, self.kernel_width,self.kernel_height, depth)
        self.biases = np.random.randn(num_kernels)

    def convolve(self, img, kernels):
        if len(img.shape) < 3:
            raise ValueError("Input must have at least three dimensions (height, width, depth).")
          
        x_kernel_shape = kernels.shape[2] 
        y_kernel_shape = kernels.shape[1] 
        
        x_img_shape = img.shape[0] 
        y_img_shape = img.shape[1]

        x_output = int((x_img_shape + (2 * self.padding) - x_kernel_shape)/ self.stride) + 1
        y_output = int((y_img_shape + (2 * self.padding) - y_kernel_shape)/ self.stride) + 1
        
        if self.padding != 0:
            image_padded = np.pad(img, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant', constant_values=0)
        else:
            image_padded = img

        #(lines, columns, depth)
        output = np.zeros((x_output, y_output, kernels.shape[3]))

        #(for every kernel)
        for k in range(kernels.shape[0]):
            #(current kernel and bias)
            kernel = kernels[k]
            bias = self.biases[k]
            #(for x and y in ouput)
            for y in range(y_output):
                for x in range(x_output):
                    #select the window of correlation, sum it * the kernel and add the bias
                    x_start = x * self.stride
                    x_end = x_start + x_kernel_shape
                    y_start = y * self.stride
                    y_end = y_start + y_kernel_shape
                    img_slice = image_padded[x_start:x_end, y_start:y_end, :]
                    output[x, y, k] = np.sum(img_slice * kernel) + bias
        return output
    
    def forward(self, input):
        self.input = input
        self.output = self.convolve(input, self.kernels)
        return self.output
    
    def backward(self, dvalues):
        self.dinput = np.zeros_like(self.input)
        dx_padded = np.zeros_like(self.dinput)
        
        x_kernel_shape = self.kernels.shape[2]
        y_kernel_shape = self.kernels.shape[1]
        
        for k in range(self.num_kernels):
            kernel = self.kernels[k]
            for y in range(dvalues.shape[1]):
                for x in range(dvalues.shape[0]):
                    x_start = x * self.stride
                    x_end = x_start + x_kernel_shape
                    y_start = y * self.stride
                    y_end = y_start + y_kernel_shape
                    img_slice = self.input[x_start:x_end, y_start:y_end, :]
                    dx_padded[x_start:x_end, y_start:y_end, :] += kernel * dvalues[x, y, k]
                    
        if self.padding != 0:
            self.dinput = dx_padded[self.padding:-self.padding, self.padding:-self.padding, :]
        else:
            self.dinput = dx_padded
            
        return self.dinput



class Pooling:
    def __init__(self, pooling_width=2, pooling_height=2):
        self.pooling_width = pooling_width
        self.pooling_height = pooling_height

    def max_pooling(self, img):
        output_height = img.shape[0] // self.pooling_height
        output_width = img.shape[1] // self.pooling_width
        depth = img.shape[2]
        pooling = np.zeros((output_height, output_width, depth))
        for l in range(depth):
            for i in range(0, output_height * self.pooling_height, self.pooling_height):
                for j in range(0, output_width * self.pooling_width, self.pooling_width):
                    patch = img[i:i+self.pooling_height, j:j+self.pooling_height,l]
                    result = np.max(patch)
                    pooling[i // self.pooling_height, j // self.pooling_width, l] = result
            
        return pooling
    
    def forward(self, img):
        self.img = img 
        self.output = self.max_pooling(img)
        return self.output

    def backward(self, dvalues):
        dinput = np.zeros_like(self.img)
        output_height, output_width, depth = dvalues.shape
        
        for l in range(depth):
            for i in range(output_height):
                for j in range(output_width):
                    patch = self.img[i*self.pooling_height:(i+1)*self.pooling_height, j*self.pooling_width:(j+1)*self.pooling_width, l]
                    max_val = self.output[i, j, l]
                    mask = (patch == max_val)
                    dinput[i*self.pooling_height:(i+1)*self.pooling_height, j*self.pooling_width:(j+1)*self.pooling_width, l] = mask * dvalues[i, j, l]
                    
        return dinput


class Fully_Connected:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons));
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)



