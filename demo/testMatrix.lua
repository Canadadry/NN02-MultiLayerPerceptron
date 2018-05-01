package.path = package.path .. ";../src/?.lua"
require "Matrix"

m1 = Matrix.fromVector({1,2,3})
m1:print()
m2 = m1:transpose()
m2:print()
m3 = m1 * m2
m3:print()


m4 = m1*m2 + m1*m2
m4:print()

m33 = Matrix(3,3,1)
m33:print()
m5 = m33 * m1 + Matrix(3,1,1)
m5:print()


m5:map(function (x) return math.random()*2-1 end)
m5:print()