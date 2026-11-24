import torch
import torch.nn as nn
import torch.autograd as autograd

class DPNAutoGrad(autograd.Function):
    @staticmethod
    def forward(ctx, x, output_size, total_nodes, biases, *weights):

        input_size = x.shape[1]
        start_node = 0
        hidden_nodes = total_nodes - output_size
        for i in range(len(weights)):
            W = weights[i]
            end_node = start_node + W.shape[0]
            
            output = x.matmul(W.T) + biases[start_node:end_node]
            if start_node < hidden_nodes:  # if the current block has non-output neurons
                output = output.relu()  # apply ReLU activation
            x = torch.cat((x, output), dim=1)
            start_node = end_node

        ctx.input_size = input_size
        ctx.nodes = total_nodes
        ctx.output_size = output_size
        ctx.save_for_backward(x, *weights)

        return x[:, -output_size:]

    @staticmethod
    def backward(ctx, grad_output):

        #retrieve saved tensors
        saved = ctx.saved_tensors
        x = saved[0]
        weights = saved[1:]

        #variables used for gradient calculation
        output_size = ctx.output_size
        input_size = ctx.input_size
        nodes = ctx.nodes

        output_threshold = input_size + nodes - output_size

        d_weights = []
        
        d_output = x.new_zeros(x.shape)
        d_output[:, -output_size:] = grad_output

        output_end_idx = input_size + nodes
        for i in reversed(range(len(weights))):

            W = weights[i]

            output_start_idx = output_end_idx - W.shape[0]

            d_o = d_output[:, output_start_idx : output_end_idx]

            if output_start_idx < output_threshold: #if the current block has non-output neurons
                d_o *= (x[:, output_start_idx : output_end_idx] > 0).float() #apply relu gradient

            d_weights.append(d_o.T.matmul(x[:, :output_start_idx])) #add to the beginning of d_weights
            d_output[:,  :output_start_idx] += d_o.matmul(W)
            output_end_idx = output_start_idx

        d_weights.reverse()
        d_biases = d_output[:, input_size:].sum(0)
        d_input = d_output[:, :input_size]

        return d_input, None, None, d_biases, *d_weights

class DPN(nn.Module):
    def __init__ (self, input_features, total_nodes, output_nodes, minimal = True, hidden_dims = None):
        super(DPN, self).__init__()
        self.weights = nn.ParameterList()
        self.output_nodes = output_nodes
        self.input_features = input_features
        self.total_nodes = total_nodes
        self.use_min_weights = minimal
        self.compile(hidden_dims)

         # One mask per weight tensor
        for i, w in enumerate(self.weights):
            self.register_buffer(f"mask_{i}", torch.ones_like(w))

    def compile(self, dims):
        if dims == None:
            weight_list, biases = DPN.get_weights(self.total_nodes, self.input_features, self.output_nodes, self.use_min_weights)
            self.weights.extend(weight_list)
            self.biases = biases
        else:
            weights = []
            X = self.input_features
            self.biases = biases = nn.Parameter(torch.empty(sum(dims)).uniform_(0.0, 1.0))
            for dim in dims:
                std_dev = torch.sqrt(torch.tensor(2 / X))
                weights.append(nn.Parameter(torch.randn(dim, X) * std_dev))
                X += dim
            self.weights.extend(weights)
                

    @staticmethod
    def get_weights(n, X, output_size, min_blocks):
        #n: number of neurons
        #X: input size
        #min_blocks: Bool to determine whether to calculate minimum number of possible blocks or maximum

        #CALCULATE MINIMUM BLOCKS
        if min_blocks:
            weights = []
            if n == output_size: #if all the neurons are output neurons, create one block
                std_dev = torch.sqrt(torch.tensor(2 / X, dtype=torch.float))
                weights.append(nn.Parameter(torch.randn(n, X) * std_dev))

            else: #if there are more neurons than output size, create two blocks

                std_dev = torch.tensor([2 / X, 2 / (X + n - output_size)], dtype=torch.float32)
                std_dev = torch.sqrt(std_dev)

                #First Block
                w1 = torch.randn(n - output_size, X)
                w1 *= std_dev[0]
                weights.append(nn.Parameter(w1))

                #Second Block
                weights.append(nn.Parameter(torch.randn(output_size, X + n - output_size) * std_dev[1]))

            biases = nn.Parameter(torch.empty(n).uniform_(0.0, 1.0))
            return weights, biases

        #CALCULATE MAXIMUM BLOCKS
        std_dev = torch.arange(X, X + n).view(-1, 1)
        std_dev = torch.sqrt(2 / std_dev)

        weights = []
        biases = nn.Parameter(torch.empty(n).uniform_(0.0, 1.0))

        #initialize weights for the rest of the blocks
        for i in range(n):
            weights.append(nn.Parameter(torch.randn(1, X + i) * std_dev[i]))

        return weights, biases
    
    def forward(self, x):
        
        masked_weights = [w * getattr(self, f"mask_{i}") for i, w in enumerate(self.weights)]
        return DPNAutoGrad.apply(x, self.output_nodes, self.total_nodes, self.biases, *masked_weights)

    def count_parameters(self):
        total = 0
    
        for i, w in enumerate(self.weights):
            mask = getattr(self, f"mask_{i}", None)
    
            if mask is not None:
                total += mask.count_nonzero().item()
            else:
                total += w.numel()
    
        # biases
        total += self.biases.numel()
    
        return total

    def get_mlp_dims(self):
        
        i_size = self.layers[0].in_features
        steps = len(self.layers)
        o_size = self.output_nodes
        total_params = self.count_parameters()

        #current_config = minimum configuration = all layers having the same dimension i.e output size
        current_config = [i_size]
        current_config += [o_size] * steps
        
        #current_parameters = input size * output size for the first layer and the same thing (output size * output size) for the rest of the layers plus biases
        current_params = (i_size * o_size) + (steps - 1) * (o_size * o_size) + (steps * o_size)

        for i in range(1, len(current_config) - 1):
        
            if current_config[i] - current_config[i + 1] == 0:
                current_config[i] += 1
                current_params += current_config[i + 1] + current_config[i - 1] + 1
        
                if current_params >= total_params:
                    break
                
                for j in range(1, i):
                    current_config[j] += 1
                    current_params += current_config[j + 1] + current_config[j - 1] + 1
                    if current_params >= total_params:
                        break
        
                if current_params >= total_params:
                    break

        while current_params < total_params:
            for j in range(1, len(current_config) - 1):
                current_config[j] += 1
                current_params += current_config[j + 1] + current_config[j - 1] + 1
                if current_params >= total_params:
                    break

        return current_config[1: -1]
