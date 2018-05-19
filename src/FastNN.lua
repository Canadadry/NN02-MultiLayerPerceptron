package.cpath = "../../LuaLibMatrix/build/?.so"
require "lua_matrix"

require "Class"

FastNN = class()

function FastNN:init(input,hidden,output)
	self.layer_i_h  = Matrix.new(hidden,input,0)
	self.layer_i_h:map(FastNN.randomize)
	self.layer_h_o  = Matrix.new(output,hidden,0)
	self.layer_h_o:map(FastNN.randomize)

	self.bias_h = Matrix.new(hidden,1,0)
	self.bias_h:map(FastNN.randomize)
	self.bias_o = Matrix.new(output,1,0)
	self.bias_o:map(FastNN.randomize)

	self.learningRate = 1.0

	self.input  = Matrix.new(input,1,0)
	self.hidden = Matrix.new(hidden,1,0)
	self.output = Matrix.new(output,1,0)
	self.target = Matrix.new(output,1,0)

	self.output_err = Matrix.new(output,1,0)
	self.hidden_err = Matrix.new(hidden,1,0)

	self.layer_h_o_t = Matrix.new(hidden,output,0)
	self.gradient = Matrix.new(output,1,0)

	self.hidden_t = Matrix.new(1,hidden,0)
	self.weight_ho_delta = Matrix.new(output,hidden,0)
	self.hidden_gradient = Matrix.new(hidden,1,0)
	self.input_t = Matrix.new(1,input,0)
	self.weight_ih_delta = Matrix.new(hidden,input,0)

end

function FastNN.loadVector(matrix,vector)
	matrix:map(function (i,j,v) return vector[(i+1)*(j+1)] end)
end

function FastNN:feed( input )
	if type(input) ~= 'userdata' or input.__name ~= "MatrixeMetatableName" then
		if( type(input) ==  'table' and #input == self.input:rows()) then
			self.loadVector(self.input,input)
		else
			return 
		end
	end

	Matrix.mul(self.layer_i_h,self.input,self.hidden)
	Matrix.add(self.hidden,self.bias_h)
	Matrix.map(self.hidden,FastNN.sigmoid)

	Matrix.mul(self.layer_h_o,self.hidden,self.output)
	Matrix.add(self.output,self.bias_o)
	Matrix.map(self.output,FastNN.sigmoid)
	
	return self.output
end

function FastNN:train(input,target)

	if type(target) ~= 'userdata' or target.__name ~= "MatrixeMetatableName" then
		if type(target) ==  'table' and #target == self.target:rows() then
			FastNN.loadVector(self.target,target)
		else
			return 
		end
	end

	self:feed(input)
	Matrix.copy(self.target,self.output_err)
	Matrix.sub(self.output_err,self.output)

	Matrix.transpose(self.layer_h_o,self.layer_h_o_t)
	Matrix.mul(self.layer_h_o_t,self.output_err,self.hidden_err)

	Matrix.copy(self.output,self.gradient)
	Matrix.map(self.gradient,FastNN.sigmoid_derivative)
	Matrix.hadamard_mul(self.gradient,self.output_err)
	Matrix.mulnum(self.gradient,self.learningRate)

	Matrix.add(self.bias_o,self.gradient)

	Matrix.transpose(self.hidden,self.hidden_t)
	Matrix.mul(self.gradient,self.hidden_t,self.weight_ho_delta)

	Matrix.add(self.layer_h_o,self.weight_ho_delta)

	Matrix.copy(self.hidden,self.hidden_gradient)
	Matrix.map(self.hidden_gradient,FastNN.sigmoid_derivative)
	Matrix.hadamard_mul(self.hidden_gradient,self.hidden_err)
	Matrix.mulnum(self.hidden_gradient,self.learningRate)

	Matrix.add(self.bias_h,self.hidden_gradient)

	Matrix.transpose(self.input,self.input_t)
	Matrix.mul(self.hidden_gradient,self.input_t,self.weight_ih_delta)

	Matrix.add(self.layer_i_h,self.weight_ih_delta)
end


function FastNN:serialize(filename)
	return table.save(self,filename)
end

function FastNN.deserialize(filename)
	local data  =  table.load(filename)
	local nn = NeuralNetwork(1,1,1)
	nn.layer_i_h.mtx = data.layer_i_h.mtx
	nn.layer_h_o.mtx = data.layer_h_o.mtx
	nn.bias_h.mtx = data.bias_h.mtx
	nn.bias_o.mtx = data.bias_o.mtx
	return nn
end

function FastNN.sigmoid(i,j,x)
	return 1 / (1 + math.exp(-x))
end

function FastNN.sigmoid_derivative(i,j,x)
	return x*(1-x)
end

function FastNN.randomize(i,j,x)
	return math.random()*2-1 
end

