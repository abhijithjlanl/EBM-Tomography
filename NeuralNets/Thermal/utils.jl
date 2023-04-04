using ITensors
using LinearAlgebra
using StatsBase
using DelimitedFiles
using IterTools:product
using Flux
include("ThermalMPO.jl")
include("MPS_sampler.jl")
GRISE_PATH = "../../Utils/"

include(GRISE_PATH*"MCMC.jl")



T= [1 0 ;  1/sqrt(3)   sqrt(2/3) ;
      1/sqrt(3)   sqrt(2/3) *exp(im* 2* pi /3); 1/sqrt(3)   sqrt(2/3) *exp(im* 4* pi /3) ]

     # T = [U*T[i,:] for i in 1:4]
     # T = transpose(hcat(T...))


MVec = [[T[i,j] for j in 1:2] for i in 1:4]
M = [0.5*reshape(MVec[i], 2,1)* conj(reshape(MVec[i], 1,2)) for i in 1:4]



#a = (1/24) * [6; 2; 2; 2]
#d = [a circshift(a,1)  circshift(a,2)  circshift(a,3)  ]
d = hcat([[tr(M[i]*M[j]) for i in 1:4] for j in 1:4]...)
d_inv =  inv(d)
T_mats = [ sum(d_inv[i,j]*M[j] for j in 1:4 ) for i in 1:4]




function Tetra_POVM(n , out_index ,  out_index_povm)
    #index_list here must be out_index
    T = [1 0 ;  1/sqrt(3)   sqrt(2/3) ;
 1/sqrt(3)   sqrt(2/3) *exp(im* 2* pi /3); 1/sqrt(3)   sqrt(2/3) *exp(im* 4* pi /3) ]
    POVM_LIST = []
    for i in 1:n
        push!(POVM_LIST, ITensor( T, out_index_povm[i] , out_index[i]) )
    end
    return POVM_LIST

end


function TIM_Thermal_twobody_states(n_qubits,  β, maxlinkdim = 30)
    expH, Z =  ThermalMPO(n_qubits , β, maxlinkdim = 30; rerun = true)

    one_body= Dict()
    two_body= Dict()
    for i in 1:n_qubits
        indices = deleteat!(Array(1:n_qubits) , i)
        A = [expH[j]*delta(siteinds(expH)[j]) for j in indices ]
        B =  expH[i]*reduce(*,A)/Z
        one_body[(i)] =  matrix(B)
    end

    for i in 1:n_qubits, j in  i+1:n_qubits
        A = []
        for k in 1:n_qubits
            if k == i || k == j
                push!(A, expH[k])
             else
                 push!(A , expH[k]*delta(siteinds(expH)[k] ))
             end
        end
        rho =  reduce(*,A)
        rinds = inds(rho)
        @assert rinds[1].id == rinds[2].id; "Order is wrong, 1 and 2 must be at same site"
        @assert rinds[1].plev ==  rinds[3].plev == 1; " 1 and 3, must be primed"
        newinds = [rinds[2], rinds[4], rinds[1], rinds[3] ]
        mat =  reshape( Array( rho, newinds...), 4,4)
        two_body[(i,j)] = mat/tr(mat) 
    end

    return one_body, two_body
end


function TIM_Thermal_exact_sampler(n_samples , n_qubits, β;  maxlinkdim = 30)
    q = 4
    expH, Z =  ThermalMPO(n_qubits , β, maxlinkdim = maxlinkdim; rerun = true)
    outind_index_povm = []
    for i in 1:n_qubits
           push!(outind_index_povm, Index(q; tags="Site,n=$i"))
    end
    P =  Tetra_POVM(n_qubits, [x[1] for x in siteinds(expH)] , [x[2] for x in siteinds(expH)], outind_index_povm)
    Prob_MPS = MPS([P[i]* expH[i] for i in 1:n_qubits])
    map(normalize!, Prob_MPS)
    marginal_MPS =  marginals(Prob_MPS, q)
    samples =  exact_sampler(marginal_MPS, siteinds(Prob_MPS), q,n_qubits, n_samples)
    return samples
end



function marginals(PROB_MPS, q)
    marginal_MPS = [deepcopy(PROB_MPS)]
    n =  length(PROB_MPS)
    for i in 1:n-1
        @show "Computing maginal $i"
        A =  deepcopy(PROB_MPS)
        for j in 1:i
            A[j] =  A[j]*delta(siteinds(A)[j])
        end
        B = reduce(*,A[1:i+1])
        C = [A[x] for x in i+2:n]
        #return [B , C...]
        A = MPS([B , C...])
        push!(marginal_MPS, deepcopy(A))
    end
    return marginal_MPS
end







function Amplitude_list(n, POVM_LIST, MPS_LIST)
        #Make sure that the POVM and MPS have the same
        # Indicis so that they contract properly
        #Z =  real(normalize(n, MPS_LIST).store)[1]
        #Psi =  contract(n, MPS_LIST)
        A = []
        for i in 1:n
            push!(A, MPS_LIST[i]*POVM_LIST[i])
        end

        return A
end
function simplex_proj(x)
    ### Project a vector x to the probability simplex
    sx =  sort(vec(x), rev = true)
    y = [ (sum(sx[1:k]) - 1)/k for k in 1:length(sx)]
    K =  max(findall( i -> (i == 1) , y .< sx)...)
    τ = y[K]
    return [max(a -  τ, 0) for a in x]
end

function psd_project(M)
    ##TODO Does projecting on to the simplex improve the error
    vals, vecs =  eigen(M)
    proj_vals = simplex_proj(real.(vals))
    return vecs * diagm(proj_vals) * conj(transpose(vecs))
end






σZ = [ 1 0; 0 -1 ]
σX = [0 1; 1 0]

function ZZ_basis_dict(two_body_reduced)
    ZZ_vals = deepcopy(two_body_reduced)
    for (k,v) in two_body_reduced
        if length(k) == 2
            ZZ_vals[k] = tr( kron(σZ, σZ)*v)
        elseif length(k) == 1
            ZZ_vals[k] = tr(σZ* v)
        end
    end
    return ZZ_vals
end



function XX_basis_dict(two_body_reduced)
    XX_vals = deepcopy(two_body_reduced)
    for (k,v) in two_body_reduced
        if length(k) == 2
            XX_vals[k] = tr( kron(σX, σX)*v)
        elseif length(k) == 1
            XX_vals[k] = tr(σX* v)
        end
    end
    return XX_vals
end






function t_test(x; conf_level=0.95)
    alpha = (1 - conf_level)
    tstar = quantile(TDist(length(x)-1), 1 - alpha/2)
    SE = std(x)/sqrt(length(x))
   return  tstar * SE
  end

function dict_error(rd1, rd2)
    v_arr =  [rd1[k] - rd2[k] for k  in keys(rd1)]
    k_arr =  [keys(rd1)...]
    return Dict(zip(k_arr, v_arr))
end
tr_norm(M) =  sum( abs.(eigvals(M)) )
op_norm(M) =  max( abs.(eigvals(M))... )
max_arr(x) = max(x...)

function ZZ_err(M)
    if size(M)[1] == 2
        return abs(tr(σZ* M))
        end
    if size(M)[1] == 4
        return abs(tr(kron(σZ, σZ)*M))
        end
    end

function dist_measure(one_err, two_err;aggregator = mean, mat_norm = tr_norm )
        one_err_arr = [mat_norm(one_err[k]) for k in keys(one_err) ]
        two_err_arr = [mat_norm(two_err[k]) for k in keys(two_err) ]
        return aggregator( [ one_err_arr..., two_err_arr...])
end





function NN_learn_reduced_states(n_qubits, n_samples; n_est_samples=10000, init_state = 0,
     activation_list =  [swish, swish,swish, identity], args = [])
    if length(args) == 0
        run(`python learn_nn.py $n_samples $n_qubits`)
    else 
        run(`python learn_nn.py $n_samples $n_qubits $(args[1]) $(args[2]) $(args[3]) $(args[4]) $(args[5]) $(args[6])`)
       # @show Cmd(["python learn_sample_err.py $n_samples $n_qubits "*"$args"[2:end-1]])
       # run(Cmd(["python learn_sample_err.py $n_samples $n_qubits "*"$args"[2:end-1]]) )

    end
    path_list= []
    for i in 1:n_qubits
        push!(path_list, "saved_model/model$i.h5" )
        path =  "saved_model/model$i.h5"
    end
    H =  load_model(path_list, activation_list)
    #    [Check_NN_correctness[H[i]] for i in 1:n_qubits]
    if init_state == 0
        init_state = Int.(ones(n_qubits))
    end

    est_samples = Gibbs_sampler(n_qubits,4 ,n_est_samples , 40000, H, init_state = init_state)
    (one_body, two_body) = two_body_reduced_states_from_samples(est_samples)
    return one_body, two_body
    end


function two_body_reduced_states_from_samples(est_samples)
    ##Given samples return first four body reduced states

    n_samples_est = length(est_samples)
    n_qubits =  length(est_samples[1])
    ds = countmap(est_samples)
    one_body_dist = Dict()
    two_body_dist = Dict()
    for i in 1:n_qubits
        one_body_dist[(i)] = zeros(4)
        for j in i+1:n_qubits
            two_body_dist[(i,j)] =  zeros(4,4)
        end
    end

    for (k,v) in ds
        for i in 1:n_qubits
            one_body_dist[(i)][k[i]] +=  v/n_samples_est
            for j in i+1:n_qubits
                two_body_dist[(i,j)][k[i],k[j]] += v/n_samples_est
            end
        end


     end
    one_body= Dict()
    two_body= Dict()

    A =    Base.Iterators.product(repeat([1:4],2)...)
    for i in 1:n_qubits
        one_body[(i)] =  sum(T_mats[t]*one_body_dist[(i)][t] for t in 1:4) 
        for j in i+1:n_qubits
            two_body[(i,j)] =  sum( kron(T_mats[k[1]],T_mats[k[2]])*two_body_dist[(i,j)][k[1], k[2]]  for k in A) 
        end
    end

    return one_body, two_body
end

