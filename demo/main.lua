package.path = package.path .. ";../src/?.lua"
require "NeuralNetwork"
require "serialize"

function love.load(arg)
	math.randomseed( os.time() )
	nn = NeuralNetwork(2,4,1)
	count = 0
	step = 50
	size = 10
	dataset = {
		{ input = {1,0}, output ={1} },
		{ input = {1,1}, output ={0} },
		{ input = {0,1}, output ={1} },
		{ input = {0,0}, output ={0} },
	}

	testValue = {}

	for i=1,love.graphics.getWidth(),step do		
		for j=1,love.graphics.getHeight(),step do
			data = {
				x=i+step/2,
				y=j+step/2,
				i=(i+step/2)/love.graphics.getWidth(),
				j=(j+step/2)/love.graphics.getHeight();
			}
			data.guess = nn:feed({data.i,data.j})
			table.insert(testValue,data)
		end
	end    
end

function love.update(dt)
	if love.keyboard.isDown('escape') then
		love.event.push('quit')
	end

	if love.keyboard.isDown('return') then
		for i=1,10 do
			count = count+1
			local data = dataset[math.random(1,4)]
			nn:train(data.input,data.output)
		end
		for i,v in ipairs(testValue) do
			v.guess = nn:feed({v.i,v.j})
		end
	end
end

function love.keypressed( key, scancode, isrepeat )
	if key == 's' and isrepeat == false then
		nn:serialize('save_nn.lua')
	end
	if key == 'r' and isrepeat == false then
		nn = NeuralNetwork.deserialize('save_nn.lua')
		for i,v in ipairs(testValue) do
			v.guess = nn:feed({v.i,v.j})
		end
	end
end

function love.draw(dt)
	for i,v in ipairs(testValue) do
		drawPoint(v.x,v.y,v.guess)
	end
    love.graphics.setColor({1,1,1})
	love.graphics.print(count)
end

function drawPoint(x,y,guess)
    love.graphics.setColor(guess:get(1,1) > 0.5 and {1,0,0} or {0,1,0})
    love.graphics.circle("fill", x, y, size) 
end
