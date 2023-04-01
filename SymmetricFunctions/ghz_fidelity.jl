include("GHZ_sampler.jl")
include("Sym_GRISE.jl")

n  =  5
n_samples  = 6*10^4

H = Haar(2)
U = rand(H,2)


T= [1 0 ;  1/sqrt(3)   sqrt(2/3) ;
      1/sqrt(3)   sqrt(2/3) *exp(im* 2* pi /3); 1/sqrt(3)   sqrt(2/3) *exp(im* 4* pi /3) ]

      T = [U*T[i,:] for i in 1:4]
      T = transpose(hcat(T...))


MVec = [[T[i,j] for j in 1:2] for i in 1:4]
M = [0.5*reshape(MVec[i], 2,1)* conj(reshape(MVec[i], 1,2)) for i in 1:4]


d = [[ tr(M[i]*M[j]) for i in 1:4] for j in 1:4]
d = hcat(d...)
d_inv =  inv(d)
T_mats = [ sum(d_inv[i,j]*M[j] for j in 1:4 ) for i in 1:4]

p = 0.5

##Generating samples from ρ_{n,p}
samples =  GHZ_Sampler_MCMC_prep_error(n, n_samples, U , err = p, burnin=10^6)
###Learning EBM
s = optimizer_occupationnumber(samples, lambda= 1e-7)
start = samples[rand(1:n_samples)]
##Estimating Fiedlity 
(F, Flist) = MCMC_GHZ_fidelity(n, 5*10^6, s, start, U, burnin=10^6)
println("Fidelity of EBM learned from p = $p is, ", F)


