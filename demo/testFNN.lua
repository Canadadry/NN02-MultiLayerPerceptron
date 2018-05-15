package.path = package.path .. ";../src/?.lua"
require "FastNN"

math.randomseed( os.time() )

nn = FastNN(2,2,1)

print(nn:feed({1,1}))

nn:train({1,1},{1})

