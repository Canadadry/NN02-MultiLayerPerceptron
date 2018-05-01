require "Class"

Matrix = class()

function Matrix:init( rows, columns, value )
	self.mtx      = {}
	local rows    = rows or 1
	local columns = columns or 1
	local value   = value or 0
	for i = 1,rows do
		self.mtx[i] = {}
		for j = 1,columns do
			self.mtx[i][j] = value
		end
	end
end

function Matrix:copy(  )
	assert(self:is_a(Matrix),"Arg 1 must be a matrix")
	m = Matrix(self:size())
	for i = 1,self:rows() do
		for j = 1,self:columns() do
			m.mtx[i][j] = self.mtx[i][j] 
		end
	end
	return m
end

function Matrix:transpose(  )
	assert(self:is_a(Matrix),"Arg 1 must be a matrix")
	m = Matrix(self:columns(),self:rows())
	for i = 1,self:rows() do
		for j = 1,self:columns() do
			m.mtx[j][i] = self.mtx[i][j] 
		end
	end
	return m
end

function Matrix:rows()
	assert(self:is_a(Matrix),"Arg 1 must be a matrix")
	return #self.mtx
end

function Matrix:columns()
	assert(self:is_a(Matrix),"Arg 1 must be a matrix")
	return #self.mtx[1]
end


function Matrix:size( )
	assert(self:is_a(Matrix),"Arg 1 must be a matrix")
	return self.rows(),self.columns()
end

function Matrix:get( i,j )
	assert(self:is_a(Matrix),"Arg 1 must be a matrix")
	assert(type(i)=='number',"Arg 2 must be a number")
	assert(type(j)=='number',"Arg 3 must be a number")
	if self.mtx[i] and self.mtx[i][j] then
		return self.mtx[i][j]
	end
end

function Matrix:set( i,j,value )
	assert(self:is_a(Matrix),"Arg 1 must be a matrix")
	assert(type(i)=='number',"Arg 2 must be a number")
	assert(type(j)=='number',"Arg 3 must be a number")
	assert(type(value)=='number',"Arg 4 must be a number")
	if self:get( i,j ) then
		self.mtx[i][j] = value
		return 1
	end
end

function Matrix:print()
	assert(self:is_a(Matrix),"Arg 1 must be a matrix")
	for i = 1, self:rows() do
		local str =  ""
		for j = 1,self:columns() do
			str = str .. string.format(' %03.3f ',self.mtx[i][j])
		end
		print(str);
	end
	print()
end

function Matrix:map(fn)
	assert(self:is_a(Matrix),"Arg 1 must be a matrix")
	assert(type(fn)=='function',"Arg 2 must be a function")
	for i = 1, self:rows() do
		for j = 1,self:columns() do
			self.mtx[i][j] = fn(self.mtx[i][j]) 
		end
		print(str);
	end
	print()
end

function Matrix.fromVector( vec )
	assert(type(vec)=="table","Arg 1 must be a table")
	local m = Matrix(#vec)
	for i = 1, #vec do
		m.mtx[i][1] = vec[i]
	end
	return m
end

function Matrix.add( m1, m2 )
	assert(m1:is_a(Matrix),"Arg 1 must be a matrix")
	assert(m2:is_a(Matrix),"Arg 2 must be a matrix")
	assert(m1:size() == m2:size(), "matrix dont have the same size")
	local m = Matrix(m1:size())
	for i = 1, m1:row() do
		for j = 1,m1:columns() do
			m.mtx[i][j] = m1.mtx[i][j] + m2.mtx[i][j]
		end
	end
	return m
end

function Matrix.sub( m1, m2 )
	assert(m1:is_a(Matrix),"Arg 1 must be a matrix")
	assert(m2:is_a(Matrix),"Arg 2 must be a matrix")
	assert(m1:size() == m2:size(), "matrix dont have the same size")
	local m = Matrix(m1:size())
	for i = 1, m1:row() do
		for j = 1,m1:columns() do
			m.mtx[i][j] = m1.mtx[i][j] - m2.mtx[i][j]
		end
	end
	return m
end

function Matrix.mul( m1, m2 )
	assert(m1:is_a(Matrix),"Arg 1 must be a matrix")
	assert(m2:is_a(Matrix),"Arg 2 must be a matrix")
	assert(m1:columns() == m2:rows(), "m1 columns must be equal to m2 rows")

	local m = Matrix(m1:rows(),m2:columns())
	for i = 1,m1:rows() do
		for j = 1,m2:columns() do
			local num = 0
			for n = 1, m2:rows() do
				num = num + m1.mtx[i][n] * m2.mtx[n][j]
			end
			m.mtx[i][j] = num
		end
	end
	return m
end

function Matrix.mulnum( m1, num )
	assert(m1:is_a(Matrix),"Arg 1 must be a matrix")
	assert(type(num)=='number',"Arg 2 must be a number")
	m = Matrix(m1:size())
	for i = 1,m1:rows() do
		for j = 1,m1:columns() do
			m.mtx[i][j] = m1.mtx[i][j] * num
		end
	end
	return m
end

function Matrix.__add( ... )
	return matrix.add( ... )
end

function Matrix.__sub( ... )
	return matrix.__sub( ... )
end

function Matrix.__unm( ... )
	return Matrix.mulnum( mtx,-1 )
end

function Matrix.__mul( m1, m2 )
	if type( m1 ) == 'number' then return Matrix.mulnum( m2,m1 ) end
	if type( m2 ) == 'number' then return Matrix.mulnum( m1,m2 ) end
	return Matrix.mul( m1,m2 )
end


function Matrix.__eq( m1,m2 )
	if type( m1 ) ~= type( m2 ) then
		return false
	end
	if m1:rows() ~= m2:rows() or m1:columns() ~= m2:columns() then
		return false
	end
	for i = 1,m1:rows() do
		for j = 1,m1:columns() do
			if m1.mtx[i][j] ~= m2.mtx[i][j] then
				return false
			end
		end
	end
	return true
end