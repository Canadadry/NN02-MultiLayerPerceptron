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
end

function FastNN.vec2mat(vector)
	local m = Matrix.new(#vector,1,0)
	m:map(function (i,j,v) return vector[(i+1)*(j+1)] end)
	return m
end

function FastNN:feed( input )
	if type(input) ~= 'userdata' or input.__name ~= "MatrixeMetatableName" then
		if( type(input) ==  'table') then
			input =  FastNN.vec2mat(input)
		else
			return 
		end
	end


	self.last_hidden = self.layer_i_h:mul(input):add(self.bias_h)
	self.last_hidden:map(FastNN.sigmoid)

	local output = self.layer_h_o:mul(self.last_hidden):add(self.bias_o)
	output:map(FastNN.sigmoid)

	collectgarbage()
	
	return output
end

function FastNN:train(input,target)

	if type(input) ~= 'userdata' or input.__name ~= "MatrixeMetatableName" then
		if( type(input) ==  'table') then
			input =  FastNN.vec2mat(input)
		else
			return 
		end
	end

	if type(target) ~= 'userdata' or target.__name ~= "MatrixeMetatableName" then
		if( type(target) ==  'table') then
			target =  FastNN.vec2mat(target)
		else
			return 
		end
	end

	local output = self:feed(input)
	local output_err = target:sub(output)
	local hiddent_err = self.layer_h_o:transpose():mul(output_err)
	local gradient = output:copy()
	gradient:map(FastNN.sigmoid_derivative)
	gradient = gradient:hadamard_mul(output_err):mulnum(self.learningRate)

	self.bias_o = self.bias_o:add(gradient)

	local hidden_t = self.last_hidden:transpose()
	local weight_ho_delta = gradient:mul(hidden_t)

	self.layer_h_o = self.layer_h_o:add(weight_ho_delta)

	-- hidden gradient
	local hidden_gradient = self.last_hidden:copy()
	hidden_gradient:map(FastNN.sigmoid_derivative)
	hidden_gradient = hidden_gradient:hadamard_mul(hiddent_err):mulnum(self.learningRate)

	self.bias_h = self.bias_h:add(hidden_gradient)

	local input_t = input:transpose()
	local weight_ih_delta = hidden_gradient:mul(input_t)

	self.layer_i_h = self.layer_i_h:add(weight_ih_delta)

	collectgarbage()
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

