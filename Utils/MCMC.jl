using Flux
using HDF5
using Distributions:wsample
using LinearAlgebra



struct SkipConnection
  layers
  connection  #user can pass arbitrary connections here, such as (a,b) -> a + b
end


function (skip::SkipConnection)(input)
  skip.connection(skip.layers(input), input)
end

function Base.show(io::IO, b::SkipConnection)
  print(io, "SkipConnection(", b.layers, ", ", b.connection, ")")
end

function Check_NN_correctness(NN)
    N =  length(NN)
    in_dims  =  size(NN[1].W)
    for j in 2:N
        out_dims =  size(NN[j].W)
        if out_dims[2] != in_dims[1]
            return ErrorException("Dimension mis match at layer $j")
        end
        in_dims =  out_dims
    end
end


function read_model(path, activation_fn_list)
     c =  h5open(path, "r")
     k =  keys(c)
     function splitter(y)
         a = split(y,"_")
         if length(a) == 2
             return parse(Int,a[2])
         end
         return 0
     end
     layer_list = map(x -> splitter(x), k)
     wts_biases = [ collect( values(read(a)) ) for a in c]
     layers = []
     v =  sortperm(layer_list)
     for (i,l) in enumerate(wts_biases[v])
         W =  l[1]["kernel:0"]
         b =  l[1]["bias:0"]
         push!(layers, Dense(W,b, activation_fn_list[i]))
     end

     return Chain(layers...)
 end

 function load_model(path_list, activation_fn_list)
     H = []
     for p in path_list
         push!(H, read_model(p, activation_fn_list))
     end
     return H

 end


 function read_residual_model(path, activation_fn_list)
  c =  h5open(path, "r")
  wts_biases = [   collect( values(read(a)) ) for a in c]
  layers = []
  j = 0

  for (i,l) in enumerate(wts_biases)
     if !isempty(l)
     j += 1
      W =  l[1]["kernel:0"]
      b =  l[1]["bias:0"]

          push!(layers, Dense(W,b, activation_fn_list[j]))
     end
  end

  n = length(layers)

  return Chain(SkipConnection(Chain(layers[1:n-1]...), +), layers[n] )
  end


  function load_residual_model(path_list, activation_fn_list)
      H = []
      for p in path_list
          push!(H, read_residual_model(p, activation_fn_list))
      end
      return H

  end





        Gibbs_sampler_Ising(n_variables::Int, n_samples::Int, conditional_nn_list) =  Gibbs_sampler_Ising(n_variables::Int, n_samples::Int, 1000, conditional_nn_list)




function Gibbs_sampler_Ising(n_variables::Int, n_samples::Int, burn_in::Int, cond_nn_list, lookup=false)
            state = rand([1.0,-1.0], n_variables)
            if lookup==true
                prob_lookup = Dict{Any,Any}()
            end
            if lookup==true
            for i in 1:burn_in
                    for u in 1:n_variables
                        indices = deleteat!( collect( 1:n_variables ),  u )
                        spins_u =  state[indices]

                        if (spins_u,u) in keys(prob_lookup)

                            H_u =  prob_lookup[(spins_u,u)]
                        else

                            H_u =   cond_nn_list[u]( spins_u )[1]

                            prob_lookup[(spins_u,u)] =  H_u
                        end
                        state[u] =  wsample( [1,-1],  [exp(H_u), exp(-1*H_u)])
                    end
            end
            state_list = []
            for i in 1:n_samples
                    for u in 1:n_variables
                        indices = deleteat!( collect( 1:n_variables ),  u )
                        spins_u =  state[indices]
                        if (spins_u,u) in keys(prob_lookup)
                            H_u =  prob_lookup[(spins_u,u)]
                        else
                            H_u =   cond_nn_list[u]( spins_u )[1]
                            prob_lookup[(spins_u,u)] =  H_u
                        end
                         state[u] =  wsample( [1,-1],  [exp( H_u), exp(-1*H_u)])

                    end
                    push!(state_list, copy(state))
            end
        end
                for i in 1:burn_in
                        for u in 1:n_variables
                            indices = deleteat!( collect( 1:n_variables ),  u )
                            spins_u =  state[indices]


                                H_u =   cond_nn_list[u]( spins_u )[1]


                            state[u] =  wsample( [1,-1],  [exp(H_u), exp(-1*H_u)])
                        end
                end
                state_list = []
                for i in 1:n_samples
                        for u in 1:n_variables
                            indices = deleteat!( collect( 1:n_variables ),  u )
                            spins_u =  state[indices]

                                H_u =   cond_nn_list[u]( spins_u )[1]

                             state[u] =  wsample( [1,-1],  [exp( H_u), exp(-1*H_u)])

                        end
                        push!(state_list, copy(state))
                end

                return state_list





end





Gibbs_sampler(n_variables::Int,q::Int,n_samples::Int, conditional_nn_dict) =  Gibbs_sampler(n_variables::Int, q::Int, n_samples::Int, 1000, conditional_nn_dict)


function Gibbs_sampler(n_variables::Int,q::Int ,n_samples::Int, burn_in::Int, cond_nn_dict; lookup=false, init_state = 0)
    if init_state == 0
        state =  convert.(Int, ones(n_variables) )
    else
        state =  init_state
    end
    phi(u, a) = (u==a) ? 1-1/q : -1/q
    phi_vec(s) = [phi(s,a) for a in 1:q]
    prob_lookup = Dict{Any,Any}()

    if lookup==true
    for i in 1:burn_in
            for u in 1:n_variables
                indices = deleteat!( collect( 1:n_variables ),  u )
                #spins_u =  onehotbatch( state[indices] , 1:q)
                spins_u = state[indices]
                if (spins_u,u) in keys(prob_lookup)
                    wts =  prob_lookup[(spins_u,u)]
                else
                    H_u =    cond_nn_dict[(u)]( spins_u )
                    wts =    [exp(sum( phi_vec(u).* H_u)) for u in 1:q]
                    prob_lookup[(spins_u[:],u)] =  wts[:]
                end

                state[u] =  wsample( 1:q , wts )
            end
    end
    state_list = []
    for i in 1:n_samples
            for u in 1:n_variables
                indices = deleteat!( collect( 1:n_variables ),  u )
                #spins_u =  onehotbatch( state[indices] , 1:q)
                spins_u = state[indices]
                if (spins_u,u) in keys(prob_lookup)
                    wts =  prob_lookup[(spins_u,u)]
                else
                    H_u =    cond_nn_dict[(u)]( spins_u )
                    wts =    [exp(sum( phi_vec(u).* H_u)) for u in 1:q]
                    prob_lookup[(spins_u[:],u)] =  wts[:]
                end

                state[u] =  wsample( 1:q , wts )
            end
            push!(state_list, copy(state))
    end
    return state_list
    end

    for i in 1:burn_in
            for u in 1:n_variables
                indices = deleteat!( collect( 1:n_variables ),  u )
                #spins_u =  onehotbatch( state[indices] , 1:q)
                spins_u = state[indices]

                    H_u =    cond_nn_dict[(u)]( spins_u )
                    wts =    [exp(sum( phi_vec(u).* H_u)) for u in 1:q]


                state[u] =  wsample( 1:q , wts )
            end
    end
    state_list = []
    for i in 1:n_samples
            for u in 1:n_variables
                indices = deleteat!( collect( 1:n_variables ),  u )
                #spins_u =  onehotbatch( state[indices] , 1:q)
                spins_u = state[indices]

                    H_u =    cond_nn_dict[(u)]( spins_u )
                    wts =    [exp(sum( phi_vec(u).* H_u)) for u in 1:q]

                state[u] =  wsample( 1:q , wts )
            end
            push!(state_list, copy(state))
    end
    return state_list


end



function TVD(truth::Dict{}, est::Dict{}, n_samples::Int)
    s = 0.0
    for (k ,v) in est
        if haskey(truth, k)
            s+= abs( v -  truth[k])
        else
            s+= v
        end
    end

    for (k,v) in truth
        if !haskey(est, k)
            s+=v
        end
    end





    return s/(2*n_samples)

end


function binary_fn_FT(func, K, n_vars)
    #Fourier transform of a binary function
    a =  fill([1,-1], n_vars)
    S = [collect(product(a...))...]
    f_K = 0
    for s in S
        f_K += prod( s[[K...]] )*func([s...])[1]

    end
    return f_K
end
