using StatsBase
using DelimitedFiles
using IterTools:product
using Random, RandomMatrices, LinearAlgebra, Combinatorics

T_povm = [1 0 ;  1/sqrt(3)   sqrt(2/3) ;
 1/sqrt(3)   sqrt(2/3) *exp(im* 2* pi /3); 1/sqrt(3)   sqrt(2/3) *exp(im* 4* pi /3) ]


prob(x) =  abs(x)^2


function GHZ_Sampler_MCMC_prep_error(n, n_samples, U; burnin=10000, err = 0, write_path=nothing  )
    #Prob_lookup = Dict{}()
    T= [1 0 ;  1/sqrt(3)   sqrt(2/3) ;
      1/sqrt(3)   sqrt(2/3) *exp(im* 2* pi /3); 1/sqrt(3)   sqrt(2/3) *exp(im* 4* pi /3) ]
      T = [U*T[i,:] for i in 1:4]
      T = transpose(hcat(T...))


     q =4

    function conditional_GHZ(n,u,  state,M ; pr =  err)
        indices =  deleteat!(collect(1:n), u)
        #indices =  index_list[u]
        a =  prod(M[state[i], 1] for i in indices)
        b =  prod(M[state[i], 2] for i in indices)
        return 0.5*(pr)*prob.([a*M[j,1] - b*M[j,2] for j in 1:4]) + 0.5*(1-pr)*prob.([a*M[j,1] + b*M[j,2] for j in 1:4])
    end
    state = Int.(ones(n))
    state_list = []
    for t in 1:burnin
        for i in 1:n

            p = conditional_GHZ(n, i, state, T, pr=err)
            state[i] =  wsample(1:q, p )

            #print(state)
        end
        if t%(burnin/100) < 100/burnin

            print(" Burnin- $(t/(burnin/100)) % complete \r")
        end

    end

    println("burnin complete")
    for t in 1:n_samples
        for i in 1:n
            p = conditional_GHZ(n, i, state, T, pr=err)
            state[i] =  wsample(1:q, p )
        end
        push!(state_list, state[:])
        if t%(n_samples/100) < 100/n_samples
            a =  (t/(n_samples/100))


            print(" $a % complete \r")
        end

    end
    if write_path==nothing
    else
    writedlm(write_path, state_list)
end
    return state_list
end


function GHZ_fid_expt(n, n_samples; p_list= LinRange(0,1,11), classical = false)
    cl_fid_list = []
    qu_fid_list = []

    for noise_p in p_list
      H = Haar(2)
      U = rand(H,2)
#      U = I
      T= [1 0 ;  1/sqrt(3)   sqrt(2/3) ;
          1/sqrt(3)   sqrt(2/3) *exp(im* 2* pi /3); 1/sqrt(3)   sqrt(2/3) *exp(im* 4* pi /3) ]
      T = [U*T[i,:] for i in 1:4]
      T = transpose(hcat(T...))
      MVec = [[T[i,j] for j in 1:2] for i in 1:4]
      M = [0.5*reshape(MVec[i], 2,1)* conj(reshape(MVec[i], 1,2)) for i in 1:4]
      factorials = [factorial(k) for k in 0:n ]
      q = 4
      function GHZ_weight(state)
          #indices =  index_list[u]
          a =  prod(M[i][1,1]^(state[i]) for i in 1:q)
          b =  prod(M[i][2,2]^(state[i]) for i in 1:q)
          c =  prod(M[i][1,2]^(state[i]) for i in 1:q)
          d =  prod(M[i][2,1]^(state[i]) for i in 1:q)
          println(a + b)
          println(c + d)
          coeff = multinomial(state...)
          return (0.5*(noise_p)*(a +b - c - d) + 0.5*(1-noise_p)* (a + b + c + d))*coeff
      end
      n_test_samp = 2*10^6
      samples =  GHZ_Sampler_MCMC_prep_error(n, n_samples, U , err = noise_p, burnin=10^7)
      s = optimizer_occupationnumber(samples)
      start = samples[rand(1:n_samples)]
      (F, Flist, samples_learned) = MCMC_noisy_GHZ_fidelity(n,n_test_samp, s, start, U, burnin=10^6)
      println(noise_p," " ,F)
      push!(qu_fid_list, noise_p)
      if classical
          dist1 = countmap(countmap_4.(samples_learned))
          classical_fid = 0
          for (k,v) in dist1
                classical_fid += sqrt(v*GHZ_weight(k)/n_test_samp)
          end
          println(noise_p," Classical Fid ",classical_fid)
      end
    end
end




function GHZ_fidelity_state_weight(state, T_mats)
    return 0.5*( prod( T_mats[s][1,1]   for s in state) + prod( T_mats[s][2,2]   for s in state) +
           2*real(prod( T_mats[s][2,1]   for s in state)))
end

function GHZ_fidelity_state_weight1(state_hist, T_mats,q)
    return 0.5*( prod((T_mats[s][1,1])^(state_hist[s]) for s in 1:q )   + prod((T_mats[s][2,2])^(state_hist[s]) for s in 1:q ) +
           2*real(  prod((T_mats[s][1,1])^(state_hist[s]) for s in 1:q ) ) )
end



function MCMC_GHZ_fidelity(n, n_samples, P, start, U; burnin=10^8)


    T= [1 0 ;  1/sqrt(3)   sqrt(2/3) ;
      1/sqrt(3)   sqrt(2/3) *exp(im* 2* pi /3); 1/sqrt(3)   sqrt(2/3) *exp(im* 4* pi /3) ]



    T = [U*T[i,:] for i in 1:4]
    T = transpose(hcat(T...))

    MVec = [[T[i,j] for j in 1:2] for i in 1:4]
    M = [0.5*reshape(MVec[i], 2,1)* conj(reshape(MVec[i], 1,2)) for i in 1:4]

    a = (1/24) * [6; 2; 2; 2]
    d = [a circshift(a,1)  circshift(a,2)  circshift(a,3)  ]
    d_inv =  inv(d)
    T_mats = [ sum(d_inv[i,j]*M[j] for j in 1:4 ) for i in 1:4]

    q =4
    Fid = 0.0
    state =  convert.(Int, start )
    phi(u, a) = (u==a) ? 1-1/q : -1/q
    phi_vec(s) = [phi(s,a) for a in 1:q]
    state_list = []

    F=  [ zeros(n,n,n,n) for i in 1:q]
    a =  collect(0:n-1)
    A =  fill(a, q)
    Q =  Iterators.product(A...)
    for i in 1:q

        for k in Q
            if sum(k) ==  n -1

                    F[i][(k .+1)...] =  JuMP.value(P[i][[k...]]) ###Why is there a +1 here?
            end
        end

    end
    #println( max([max(abs.(flatten(s))...) for s in F]...))
    #println(" ")
    #F = [Dictionary(D[i]) for i in 1:q]
    for t in 1:burnin
        for u in 1:n
            indices = deleteat!( collect( 1:n ),  u )
            k= countmap_4(state[indices]) .+ 1
            wts =  [ exp(F[s][k...]) for s in 1:q  ]
            state[u] =  wsample( 1:q , wts )
        end
        if t%(burnin/100) < 100/burnin
            a =  (t/(burnin/100))


            print(" Burnin- $a % complete \r")
        end

    end
    println(" ")
    println("Burnin complete")

    Fid_list = []
    for t in 1:n_samples
        for u in 1:n
            indices = deleteat!( collect( 1:n ),  u )
            k= countmap_4(state[indices]) .+ 1
            #wts =  [ exp(sum(phi_vec(s)[a]*F[a][k] for a in 1:q)) for s in 1:q  ]
            wts =  [ exp(F[s][k...]) for s in 1:q  ]
            state[u] =  wsample( 1:q , wts )
        end
        #push!(state_list, state[:])
    Fid += GHZ_fidelity_state_weight(state, T_mats)
    #Fid += GHZ_fidelity_state_weight1(countmap_4(state), T_mats,q)
        if t%(2*n_samples/100) < 100/n_samples
            a =  (t/(n_samples/100))


            print(" $a % complete \r")
            push!(Fid_list, Fid/t)
        end

    end
    return (Fid/n_samples, Fid_list)

end



