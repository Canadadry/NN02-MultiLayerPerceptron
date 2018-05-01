package.path = package.path .. ";../src/?.lua"
require "NeuralNetwork"

math.randomseed( os.time() )

nn = NeuralNetwork(2,2,1)

nn:feed({1,1}):print()