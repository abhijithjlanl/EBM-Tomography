include("/home/abhijith/Dropbox/Code/GRISE_code/Potts_learning.jl")


#Make model
n =10
q = 2
J =  rand()
println(J)

Terms = Dict{}()
M = [ 1 -1 ; -1 1]
for i in 1:n-1
    Terms[(i, i+1 )] =  J*M
end


F = FactorGraph([2],n,q,Terms)
state = rand(1:2, n)
println(state)
println(conditional_energy(2, state,F, true))
