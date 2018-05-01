package.path = package.path .. ";../src/?.lua"
require "Matrix"

m1 = Matrix.fromVector({1,2,3})
m1:print()
m2 = m1:transpose()
m2:print()
m3 = m1 * m2
m3:print()
