using ITensors
using ITensors.HDF5

function mult(D1::MPO, D2::MPO; maxlinkdim::Int64 = 20)
    D1 =  prime(D1)
    D3 =  contract(D1, D2; maxdim =  maxlinkdim)
    D3 =  setprime(siteinds, uniqueinds, D3, D2, 1)
    return D3

end

function ThermalMPO(N::Int, β::Real; τ::Real = 0.001, maxlinkdim = 20, rerun = true, blocksize =  1)
    nsteps =  β/τ
    @assert nsteps ≈ round(nsteps)
    nsteps =  Int(round(nsteps))
    if  rerun
        run(`./thermal  $N $τ`)
    end
    
    f = h5open("test.h5","r")
    global D = NDTensors.dense(read(f,  "expH", MPO))
    global Db =  deepcopy(D)
    nblocks =  Int(nsteps//blocksize)

    for _ in 1:blocksize-1
        global Db =  mult(Db, D, maxlinkdim = maxlinkdim)
    end
    
    global  Dc =  deepcopy(Db)
    Dc_list = [deepcopy(Db)]
    digits_arr =  digits(nblocks, base = 2)

    for ind in 1:length(digits_arr) - 1
        global Dc = mult(Dc, Dc, maxlinkdim= maxlinkdim)
        push!(Dc_list, deepcopy(Dc))
    end

    Dc_list = [ x[2] for x  in zip(digits_arr, Dc_list ) if x[1] != 0  ]

    Dc =  reduce(mult, Dc_list)
    Z  = [Dc[i]*delta(siteinds(Dc)[i]) for i in 1:N]
    Z = Array(contract(Z...))[1]
    return Dc, Z

end

function Tetra_POVM(n , site_index_1, site_index_2 ,  out_index_povm)
    q = 4
    #index_list here must be out_index
    T = [1 0 ;
         1/sqrt(3)   sqrt(2/3) ;
         1/sqrt(3)   sqrt(2/3) *exp(im* 2* pi /3);
         1/sqrt(3)   sqrt(2/3) *exp(im* 4* pi /3) ]
    MVec = [[T[i,j] for j in 1:2] for i in 1:q]
    M = [0.5*reshape(MVec[i], 2,1)* conj(reshape(MVec[i], 1,2)) for i in 1:q]
    A =  zeros(ComplexF64,q,2,2)
    [A[i,:,:] =  M[i] for i in 1:q]
    POVM_LIST = []
    for i in 1:n
        push!(POVM_LIST, ITensor( A, out_index_povm[i] , site_index_1[i], site_index_2[i]) )
    end
    return POVM_LIST

end

function Pauli_POVM(n , site_index_1, site_index_2 ,  out_index_povm)
    #index_list here must be out_index
    q = 6

    T = [1 0 ;
         1/sqrt(2) 1/sqrt(2); 
         1/sqrt(2) im/sqrt(2);
         0 1 ;
         1/sqrt(2) -1/sqrt(2);
         1/sqrt(2) -im/sqrt(2)]

    MVec = [[T[i,j] for j in 1:2] for i in 1:q]
    M = [(1/3)*reshape(MVec[i], 2,1)* conj(reshape(MVec[i], 1,2)) for i in 1:q]
    M[4] = M[4] + M[5] + M[6]
    M = M[1:4]
    q = 4
    A =  zeros(ComplexF64,q,2,2)
    [A[i,:,:] =  M[i] for i in 1:q]
    POVM_LIST = []
    for i in 1:n
        push!(POVM_LIST, ITensor( A, out_index_povm[i] , site_index_1[i], site_index_2[i]) )
    end
    return POVM_LIST

end
