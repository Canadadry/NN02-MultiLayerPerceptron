require "Class"
require "Matrix"
require "serialize"

NeuralNetwork = class()

function NeuralNetwork:init(input,hidden,output)
	self.layer_i_h  = Matrix(hidden,input)
	self.layer_i_h:map(NeuralNetwork.randomize)
	self.layer_h_o  = Matrix(output,hidden)
	self.layer_h_o:map(NeuralNetwork.randomize)

	self.bias_h = Matrix(hidden)
	self.bias_h:map(NeuralNetwork.randomize)
	self.bias_o = Matrix(output)
	self.bias_o:map(NeuralNetwork.randomize)

	self.learningRate = 1.0
end

function NeuralNetwork:feed( input )
	if NeuralNetwork.is_a(input,Matrix) == false then
		input = Matrix.fromVector(input)
	end

	self.last_hidden = self.layer_i_h * input + self.bias_h
	self.last_hidden:map(NeuralNetwork.sigmoid)

	local output = self.layer_h_o * self.last_hidden + self.bias_o
	output:map(NeuralNetwork.sigmoid)

	return output
end

function NeuralNetwork:train(input,target)

	if NeuralNetwork.is_a(input,Matrix) == false then
		input = Matrix.fromVector(input)
	end
	if NeuralNetwork.is_a(target,Matrix) == false then
		target = Matrix.fromVector(target)
	end

	local output = self:feed(input)
	local output_err = target - output
	local hiddent_err = self.layer_h_o:transpose()*output_err
	local gradient = output:copy():map(NeuralNetwork.sigmoid_derivative)
	gradient = Matrix.hadamard_mul(gradient,output_err)*self.learningRate

	self.bias_o = self.bias_o + gradient

	local hidden_t = self.last_hidden:transpose()
	local weight_ho_delta = gradient * hidden_t

	self.layer_h_o = self.layer_h_o + weight_ho_delta

	-- hidden gradient
	local hidden_gradient = self.last_hidden:copy():map(NeuralNetwork.sigmoid_derivative)
	hidden_gradient = Matrix.hadamard_mul(hidden_gradient,hiddent_err)*self.learningRate


	self.bias_h = self.bias_h + hidden_gradient

	local input_t = input:transpose()
	local weight_ih_delta = hidden_gradient * input_t

	self.layer_i_h = self.layer_i_h + weight_ih_delta

end


function NeuralNetwork:serialize(filename)
	return table.save(self,filename)
end

function NeuralNetwork.deserialize(filename)
	local data  =  table.load(filename)
	local nn = NeuralNetwork(1,1,1)
	nn.layer_i_h.mtx = data.layer_i_h.mtx
	nn.layer_h_o.mtx = data.layer_h_o.mtx
	nn.bias_h.mtx = data.bias_h.mtx
	nn.bias_o.mtx = data.bias_o.mtx
	return nn
end

function NeuralNetwork.sigmoid(x)
	return 1 / (1 + math.exp(-x))
end

function NeuralNetwork.sigmoid_derivative(x)
	return x*(1-x)
end

function NeuralNetwork.randomize(x)
	return math.random()*2-1 
end

