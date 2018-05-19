package.path = package.path .. ";../src/?.lua"
require "FastNN"

math.randomseed(os.time()  )


dataset = {
	{ input = {1,0}, output ={1} },
	{ input = {1,1}, output ={0} },
	{ input = {0,1}, output ={1} },
	{ input = {0,0}, output ={0} },
}

nn = FastNN(2,4,1)

for i=1,10000 do
	local data = dataset[math.random(1,4)]
	nn:train(data.input,data.output)
end

print(nn:feed({1,1}))
print(nn:feed({0,0}))
print(nn:feed({0,1}))
print(nn:feed({1,0}))