using JLD2
using CSV
using PyPlot
include("utils.jl")

using DataFrames
using ProgressBars


nq = 30
n_samples = 10^5
β = 1.0
samples =  TIM_Thermal_exact_sampler(n_samples, nq, β, maxlinkdim = 40)
one_states, two_states =  TIM_Thermal_twobody_states(nq, β)
ZZ_exact = ZZ_basis_dict(two_states)
XX_exact = XX_basis_dict(two_states)
writedlm("samples.csv", samples)
one_body, two_body = NN_learn_reduced_states(nq, n_samples; n_est_samples=10000)
map!( psd_project, values(one_body))
map!( psd_project, values(two_body))
run(`rm samples.csv`)
ZZ_learned = ZZ_basis_dict(two_body)
XX_learned = XX_basis_dict(two_body)

exact1 = []
exact2 = []
learned1 = []
learned2 = []

for j in 1:nq
    mid = 15
    if j == mid
        push!(exact1, 1.0)
        push!(learned1, 1.0)

    else
        push!(exact1, ZZ_exact[( min(mid,j), max(mid,j)  )])
        push!(learned1, ZZ_learned[( min(mid,j), max(mid,j)  )])
    end
    
end
for j in 1:nq
    mid = 1
    if j == mid
        push!(exact2, 1.0)
        push!(learned2, 1.0)

    else
        push!(exact2, XX_exact[( min(mid,j), max(mid,j)  )])
        push!(learned2, XX_learned[( min(mid,j), max(mid,j)  )])
    end

end



plot(exact1, color="red",  label="ZZ_exact")
plot(exact2,  color="blue",label="XX_exact")
plot(learned1, "o", color="red",  label="ZZ_learned")
plot(learned2,"o" ,  color="blue", label="XX_learned")
legend(loc = 5)

