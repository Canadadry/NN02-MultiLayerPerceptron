require "Class"
require "Matrix"

NeuralNetwork = class()

function NeuralNetwork:init(input,hidden,output)
	self.layer_i_h  = Matrix(hidden,input)
	self.layer_i_h:map(function (x) return math.random()*2-1 end)
	self.layer_h_o  = Matrix(output,hidden)
	self.layer_h_o:map(function (x) return math.random()*2-1 end)

	self.bias_h = Matrix(hidden)
	self.bias_h:map(function (x) return math.random()*2-1 end)
	self.bias_o = Matrix(output)
	self.bias_o:map(function (x) return math.random()*2-1 end)
end

function NeuralNetwork:feed( input )
	if NeuralNetwork.is_a(input,Matrix) == false then
		input = Matrix.fromVector(input)
	end

	local hidden = self.layer_i_h * input + self.bias_h
	hidden:map(function (x) return 1 / (1 + math.exp(-x)) end)
	local output = self.layer_h_o * hidden + self.bias_o
	output:map(function (x) return 1 / (1 + math.exp(-x)) end)

	return output

end
