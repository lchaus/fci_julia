### A Pluto.jl notebook ###
# v0.19.22

#> [frontmatter]

using Markdown
using InteractiveUtils

# ╔═╡ d050a7f8-c4c1-11ed-3542-7dffdbf79b0f
begin
	using Fermi
	using Fermi.Integrals
	using LinearAlgebra
	using BenchmarkTools
	using Combinatorics
	using OMEinsum
	using Printf
end


# ╔═╡ 2f904006-4f00-4046-b78d-e9792acb5b96
# ╠═╡ skip_as_script = true
#=╠═╡
md" ## Hartree-Fock Algorithm 
*Using the notation of Szabo & Ostlund - Introduction to Advanced Electronic Structure Theory*
"
  ╠═╡ =#

# ╔═╡ d415f46c-b803-4f69-9290-b7e722606660
begin
	#1. Specify a molecule and basis set (using fermi.jl)
	@molecule {
	    O   0.000000     0.00000     0.11779
	    H   0.000000     0.75545    -0.47116
	    H   0.000000    -0.75545    -0.47116 
	}
	@set {
	    basis sto-3g
	    charge 0
	    multiplicity 1 
	}
	
	#2. Calculate all the required molecular integrals (using fermi.jl)
	ao_integrals = IntegralHelper(eri_type=Chonky())
	S = ao_integrals["S"]
	Hcore = Array(ao_integrals["T"]) + Array(ao_integrals["V"])
	V_nuc = ao_integrals.molecule.Vnuc
	μνλσ = Array(ao_integrals["ERI"])
	
	#Store basis set size K and number of occupied orbitals nocc
	K = size(S, 1)
	nocc = ao_integrals.molecule.Nα
	
	#3. Diagonalize the overlap matrix S and obtain a transformation matrix (3.167)
	s, U = eigen(S)
	s = Diagonal(s)
	X = U * inv(sqrt(s)) * U'
	
	#4. Obtain a guess at the density matrix Pnew
	C = zeros(K,K)
	occupied = Diagonal(vcat(fill(2, nocc), zeros(K-nocc)))
	Pnew = C * occupied * C'
	
	#5. Calculate matrix G of eq (3.154) (in two steps here)
	G = μνλσ - 0.5 * permutedims(μνλσ, (1,4,3,2))
	
	iteration = 1
	Ptol = 10
	maxiter = 50
	while Ptol > 1e-4
	    P = Pnew
	    Gμν = ein"λσ, μνλσ -> μν"(P,G)
	    
	    #6. Add G to the core Hamiltonian to obtain the Fock matrix
	    F = Hcore + Gμν
	
	    #7. Calculate the transformed Fock matrix Ft
	    Ft = X' * F * X
	    
	    #8. Diagonalize Ft to obtain Ct and ε
	    ε, Ct = eigen(Ft, sortby = x->x)
	
	    #9. Calculate C
	    C = X * Ct
	
	    #10. Form a new density matrix P from C using (3.145)
	    Pnew = C * occupied * C'
	    
	    #11. Determine whether the procedure has converged
	    Ptol = sqrt.(sum((Pnew .- P).^2) ./ K^2)
	
	    E = ein"νμ,μν -> "(Pnew,(Hcore+F))
	    E = 0.5 .* E .+ V_nuc
	    println("Iteration $iteration ---- E = $E ---- Density SD $(@sprintf("%.8f", Ptol))")
	    if iteration > maxiter
	        println("Max number of iterations reached")
	        break
	    end
	    iteration += 1
	end
	println("Hartree Fock converged")
end

# ╔═╡ c6c9efc5-2ee4-4de1-8018-ac90d167c655
md"
## Full CI
N-resolution method, code adapted in Julia from pyscf:\
<https://github.com/pyscf/pyscf/blob/master/pyscf/fci/fci_slow.py> \
Following the notation of *Molecular Electronic-Structure Theory by Helgaker, Jorgensen, Olsen* \
We will use the alpha and beta string representation of the CI expansion

$\ket{I_\alpha I_\beta} = \hat{\alpha}_{I_\alpha}\hat{\beta}_{I_\beta} \ket{vac}$
$I_{\alpha} = 00000111 = 7$"

# ╔═╡ f6d71eef-2450-4612-81d5-43fb46c6fb1b
md"### Backtracking algorithm to generate  all occupation bitstrings
"

# ╔═╡ f997e31f-3844-4f84-b014-d211ab4f5997
begin
	nmo = 7
	nα = nβ = 5
	nvir = nmo - nα
	orbital_list = collect(0:nmo-1)
	
	function generate_string_iter(orbital_list,nelec)
		if nelec == 1
		    result = [(1 << i) for i in orbital_list]
		elseif nelec >= length(orbital_list)
			n = 0
			for i in orbital_list
				n = n | (1 << i)
			end
			result = [n]
		else
			restorb = orbital_list[1:end-1]
			thisorb = 1 << orbital_list[end]
			result = generate_string_iter(restorb, nelec)
			for n in generate_string_iter(restorb, nelec-1)
				tmp = (n | thisorb)
				push!(result, tmp)
			end
		end
		return result
	end
	
	α_string = generate_string_iter(orbital_list,nα)
	α_bitstring = [bitstring(i) for i in α_string]
	
	display(α_string)
	display(α_bitstring)
end

# ╔═╡ 6c5fdf02-d0f3-4471-8e27-30ee26036bd1
md"
We want to solve the eigenvalue problem using iterative methods. For this purpose we have to be able to calculate the matrix elements of the Hamiltonian efficiently. We need an efficient addressing scheme to map the monoexcited occupation strings and calculate only non-zero matrix elements of the Hamiltonian.

We will need two more functions before being able to generate a look up table containing all monoexcited occupation strings.

"

# ╔═╡ e94cc0cc-016a-4cbd-9dfe-b98d9f012556
md" ### Define the unique adress of a given excited string
"

# ╔═╡ 8afd6650-e4ca-4a58-8aec-ed31d8dcdb7f
begin
	function string2address(list_string, count, nmo, nel)
	    address_map = Vector{Int64}(undef, count)
	    nextaddr0 = binomial(nmo-1, nel)
	    for i in 1:count
	        string = list_string[i]
	        addr = 1
	        nelec_left = nel
	        nextaddr = nextaddr0
	        for norb_left=nmo-1:-1:0
	            if ( (nelec_left == 0) || norb_left < nelec_left )
	                break
	            elseif (((1 << norb_left) & string) != 0)
	            @assert nextaddr == binomial(norb_left,nelec_left)
	            addr += nextaddr
	            nextaddr *= nelec_left
	            nextaddr /= norb_left
	            nextaddr = floor(Int,nextaddr)
	            nelec_left -= 1
	            else
	                nextaddr *= norb_left - nelec_left
	                nextaddr /= norb_left
	                nextaddr = floor(Int,nextaddr)
	            end
	        end
	        address_map[i] = addr
	    end
	    return address_map
	end
	
	#List of all excited strings from ground state string
	excited_string_int = [62, 94, 61, 93, 59, 91, 55, 87, 47, 79]
	excited_string = [bitstring(i) for i in excited_string_int]
	println("Ground state string α_string[1] = \n $(bitstring(α_string[1]))")
	println("1st monoexcitation excited_string[1] = \n $(excited_string[1])")
	
	stringaddress = string2address(excited_string_int,nα*nvir,nmo,nα)
	
	println("Index of excited_string[1] in α_string is stringaddress[1] = $(stringaddress[1]) ")
	println("α_string[6] = \n $(bitstring(α_string[6]))")
end

# ╔═╡ 9149e499-2d7f-4d53-a7a3-8495cf8c3aa1
md" 
### Determine the parity of the excited string
Due to anticommutation relations of creation and annihilation operators and the number of permutations required to put a given excited string in canonical order, we need to determine for each excited string a parity factor.

$${\{a_p^\dagger, a_q\}} = a_p^\dagger a_q+ a_q a_p^\dagger = \delta_{pq}$$
$${\{a_p^\dagger, a_q^\dagger\}} = 0$$
$${\{a_p, a_q\}} = 0$$

"

# ╔═╡ 0a6f1c6c-f4c0-4ce0-8848-240eaab58839
begin
	# Returns the parity of a given "a†(p) a(q)|string>"
	function string_parity(p, q, string)
	    if (p > q)
	        mask = (1 << (p-1)) - (1 << q)
	    else
	        mask = (1 << (q-1)) - (1 << p)
	    end
	    if (count_ones(mask & string) % 2 == 1)
	        parity = -1
	    else 
	        parity = 1
	    end
	    test = (string[1] & mask) 
	    return parity
	end
	
	parity_62_string1 = string_parity(6, 2, α_string[1])
	
	println("Ground state string α_string[1] = \n $(bitstring(α_string[1]))")
	println("Parity for a†⁶a²|a¹a²a³a⁴a⁵> = $parity_62_string1" )
	
end

# ╔═╡ 9b28dc61-6da2-4230-aa26-aa57ddd97315
md" 
### Generate the monoexcitations $E_{pq} | string>$
"

# ╔═╡ afd13da3-8825-41c5-a897-41234fdd89ea
begin
	function E_pq_string_index(strings, nmo, nocc)
	    #Create look-up table of the effect of E_pq on a give string list
	    # table = [a^†, a, address, parity]
	    nvir = nmo - nocc
	    allowed = nocc + nocc * nvir
	    nstring = binomial(nmo,nocc)
	    occupied = Vector{Int64}(undef, nocc)
	    virtual = Vector{Int64}(undef, nvir)
	    table = Array{Int64}(undef, allowed, 4, nstring)
	    for (nstr, string) in enumerate(strings)
	        o = 1
	        v = 1
	        #Mapping of creation and annihilation operators
	        for i in 0:nmo-1
	            if (string & (1 << i) != 0)
	                occupied[o] = i + 1
	                o += 1
	            else 
	                virtual[v] = i + 1
	                v += 1
	            end
	        end
	
	        for i in 1:nocc
	            table[i,1,nstr] = occupied[i]
	            table[i,2,nstr] = occupied[i]
	            table[i,3,nstr] = nstr
	            table[i,4,nstr] = 1
	        end
	        
	        k = nocc + 1
	        list_string = Vector{Int64}(undef,allowed-nocc)
	        for i in 1:nocc
	            for j in 1:nvir
	                #Generate all excitations from a given "string" in "list_string"
	                string_ex = (string ⊻ ( 1 << (occupied[i]-1)) | (1 << (virtual[j] - 1)))
	                list_string[k - nocc] = string_ex
	                table[k,1,nstr] = virtual[j]
	                table[k,2,nstr] = occupied[i]
	                table[k,4,nstr] = string_parity(virtual[j], occupied[i], string)
	                k += 1
	            end
	            address = string2address(list_string,nocc*nvir,nmo,nocc)
	            for k in 1:(nocc*nvir)
	                table[k+nocc,3,nstr] = address[k]
	            end
	        end
	    end
	    return table
	end
	
	address_table = E_pq_string_index(α_string, nmo, nα)
	println("α_bitstring[1] = \n $(α_bitstring[1])")
	println("Corresponding entry -> address_table[1]")
	println("[a†, a, address, parity]")
	display(address_table[:,:,1])
end

# ╔═╡ b427df39-c75f-41a4-b001-04311d1e62fb
md" Finally we make sure to store all orbital occupation in every possible string for later accessing the molecular orbitals"

# ╔═╡ b7351384-f2bb-4a29-95aa-0e588c9b48e8
begin
	function occupied_strings(orblist, nel)    
	    if (nel == 1)
	        res = [[i] for i in orblist]
	    elseif (nel >= length(orblist))
	        res = [orblist] 
	    else
	        restorb = orblist[1:end-1]
	        thisorb = orblist[end]
	        res = occupied_strings(restorb, nel)
	        for i in occupied_strings(restorb, nel-1)
	            tmp = vcat(i, thisorb)
	            push!(res, tmp)
	        end
	    end
	    return res
	end
	
    display(occupied_strings(orbital_list, nα))
end

# ╔═╡ 06f727d6-092b-4256-9b3d-478fbcbf5671
md" And we put it all together..."

# ╔═╡ 44d0daad-87f7-4c09-a529-78ec4bbebf0b
function generate_index(orbital_list, nmo, nel)
    strings =  generate_string_iter(orbital_list,nel)
    address_table = E_pq_string_index(strings, nmo, nel)
    return address_table
end

# ╔═╡ c2920c78-3393-47d5-a9d8-100e7e0cd7ef
md" 
### The N-resolution method
Our goal is to solve for \
$$\hat{H} = \sum_{pq}h_{pq} E_{pq} + \frac{1}{2} \sum_{pqrs}g_{pqrs} E_{pq} E_{rs}$$

To solve iteratively the resulting eigenvalue problem (using Davidson algorithm) we need to be able to define a vector contraction of the Hamiltonian $\sigma = HC$

The sigma vector in the alpha/beta string basis can be split and computed in two parts (one electron part $\sigma_{I_\alpha I_\beta}^{(1)}$ and a two electron part $\sigma_{I_\alpha I_\beta}^{(2)})$

$\sigma_{I_\alpha I_\beta} = \sigma_{I_\alpha I_\beta}^{(1)} +  \sigma_{I_\alpha I_\beta}^{(2)}$

We first introduce effective one-electron integrals by absorbing the $g_{prrq}$  terms of the two-electron integral tensor.

$$k_{pq} = h_{pq} -\frac{1}{2} \sum_{r}^{N}g_{prrq} $$

$$\sigma_{I_\alpha I_\beta}^{(1)} = \sum_{pq}\sum_{J_\alpha J_\beta}k_{pq} \bra{I_\alpha I_\beta}E_{pq}\ket{J_\alpha J_\beta} C_{J_\alpha J_\beta}$$


"

# ╔═╡ 3ea77dfa-6f24-4f7b-92f6-d02e22aebe1f
function absorb_h1e(hcore, C, eri, nmo, nel)
    h2e = copy(eri)
    h1e = C' * hcore * C
    @ein  jiik[j,k] := eri[j,i,i,k]
    f1e = h1e - jiik * 0.5
    f1e = f1e * (1. / (nel+1e-100))
    for k in 1:nmo
        h2e[:,:,k,k] .+= f1e
        h2e[k,k,:,:] .+= f1e
    end
    return 0.5 * h2e
end

# ╔═╡ 8a57eead-590d-4eab-b27e-5be826de3005
md"Then we compute the two-electron part obtained by inserting the resolution of the identity.
$$\sigma_{I_\alpha I_\beta}^{(2)} = \frac{1}{2} \sum_{K_\alpha K_\beta J_\alpha J_\beta} \sum_{pqrs} \bra{I_\alpha I_\beta}E_{pq}\ket{K_\alpha K_\beta} g_{pqrs} \bra{K_\alpha K_\beta}E_{pq}\ket{J_\alpha J_\beta} C_{J_\alpha J_\beta}$$

"

# ╔═╡ fe2b36e9-3a68-4a0c-9549-79e0a0fddb73
function contract2e(eri,ci0,nmo,nel)
    #RHF case only for now nα = nβ = nel // 2
    nstrings = binomial(nmo,nel)
    allowed = nel + (nmo - nel) * nel
    #Generate bitstring tables
    string_index_α = generate_index(collect(0:nmo-1), nmo, nel)
    string_index_β = generate_index(collect(0:nmo-1), nmo, nel)

    fcivec = reshape(ci0, (nstrings,nstrings))
    fcitensor = zeros(Float64, nstrings, nstrings, nmo, nmo)
    for i in axes(string_index_α, 3)
        str = view(string_index_α,:,:, i)
        for j in axes(str,1)
            c,a,adr,sgn = str[j,1],str[j,2],str[j,3],str[j,4]
            fcitensor[adr,:,a,c] += sgn * fcivec[i,:]
        end
    end

    for i in axes(string_index_β, 3)
        str = view(string_index_β,:,:, i)
        for j in axes(str,1)
            c,a,adr,sgn = str[j,1],str[j,2],str[j,3],str[j,4]
            fcitensor[:,adr,a,c] += sgn * fcivec[:,i]
        end    
    end
    @ein fcitensor[A,B,b,j] := eri[a,i,b,j] * fcitensor[A,B,a,i]
    cinew = zeros(nstrings, nstrings)

    for i in axes(string_index_α, 3)
        str = view(string_index_α,:,:, i)
        for j in axes(str,1)
            c,a,adr,sgn = str[j,1],str[j,2],str[j,3],str[j,4]
            cinew[adr,:] += sgn * fcitensor[i,:,a,c]
        end
    end
    for i in axes(string_index_β, 3)
        str = view(string_index_β,:,:, i)
        for j in axes(str,1)
            c,a,adr,sgn = str[j,1],str[j,2],str[j,3],str[j,4]
            cinew[:,adr] += sgn * fcitensor[:,i,a,c]
        end
    end
    return cinew
end


# ╔═╡ e5fef9ac-4cd0-47a7-a3c6-e44e3be71cdd
md" For the Davidson algorithm, we also need to build a diagonal hamiltonian to use as a preconditionner

"

# ╔═╡ a3753425-2ca4-4759-ad9c-cdb14571907f
function make_hdiag(h1e, eri, nmo, nel)
    # α != β not handled
    occ_α = occupied_strings(collect(1:1:nmo),nel)
    occ_β = occupied_strings(collect(1:1:nmo),nel)
    jdiag = ein"iijj -> ij"(eri)
    kdiag = ein"ijji -> ij"(eri)
    hdiag = []
    for α in occ_α
        for β in occ_β
            e1 = sum([h1e[i,i] for i in α]) + sum([h1e[i,i] for i in β])
            e2 = sum(jdiag[α,:][:,α]) + sum(jdiag[α,:][:,β]) + sum(jdiag[β,:][:,α]) + sum(jdiag[β,:][:,β]) -sum(kdiag[α,:][:,α])- sum(kdiag[β,:][:,β])
            push!(hdiag, (e1 + (0.5 * e2)))
        end
    end
    return hdiag
end

# ╔═╡ ebf029cf-b7af-47ac-9b48-ca303e136f48
md"
### Davidson algorithm
"

# ╔═╡ 1c4f0b30-4dff-4f06-83e2-93307f3769c8
begin
	function davidson(sigma, ci0, preconditionner, nroots=2, tol=1e-12, maxiter = 50, trial_space = 12)
	    space = 0
	    trial_space = trial_space + (nroots-1) * 3
	    toloose = sqrt(tol)
	    lindep = 1e-14
	    nstrings = size(ci0)[1]
	    dimt = nroots
	    fresh_start = true
	    v = Float64[]
	    w = Float64[]
	    e = Array{Float64}(undef, nroots)
	    conv = [false for i in collect(1:1:nroots)]
	    xt = Array{Float64}(undef,nstrings,nroots)
	    axt = similar(xt)
	    xs = Array{Float64}(undef,nstrings,nroots+trial_space)
	    x0 = similar(xs)
	    ax = similar(xs)
	    ax0 = similar(xs)
	    dx_norm = Array{Float64}(undef,nroots)
	
	    for i in 1:dimt
	        x0[:,i] = ci0[:,i]
	    end
	    
	    heff = zeros((trial_space + nroots),(trial_space + nroots))
	    norm_min = 1
	    for it in 1:maxiter
	        if fresh_start == true
	            space = 1
	            dimt = nroots
	            xt = Array{Float64}(undef,nstrings,nroots)
	            xs = Array{Float64}(undef,nstrings,nroots+trial_space)
	            ax = similar(xs) 
	            xt[:,1:dimt], dimt = orthonormalise(x0[:,1:dimt],dimt,lindep)
	            if dimt != nroots
	                println("Warning - QR decomposition removed $(nroots - dimt) vectors ")
	            end
	            x0 = similar(xs) 
	        elseif space > 1
	            xt[:,1:dimt], dimt  = orthonormalise(xt[:,1:dimt], dimt, lindep)
	        end
	
	        for i in 1:dimt
	            axt[:,i] = sigma(xt[:,i])
	        end
	
	        for (i,k) in enumerate(collect(space:1:(dimt+space-1)))
	            xs[:,k] = xt[:,i]
	            ax[:,k] = axt[:,i]
	        end
	
	        rnow = dimt
	        head, space = space, space + rnow
	        elast = copy(e)
	        vlast = copy(v)
	        conv_last = copy(conv)
	        fill_heff!(heff,view(xs,:,1:space-1),view(ax,:,1:space-1), xt, axt, dimt)
	        xt =  Array{Float64}(undef,nstrings,nroots)
	        axt = similar(xt) 
	
	        w, v = eigen(heff[1:space-1,1:space-1])
	        
	        e = w[1:nroots]
	        v = v[:,1:nroots]
	
	        x0 = Array{Float64}(undef,nstrings,nroots+trial_space)
	        gen_x0(x0, v, view(xs,:,1:space-1))
	        gen_x0(ax0, v, view(ax,:,1:space-1))
	        
	        elast, conv_last = sort_elast(elast, conv_last, vlast, v, fresh_start)
	        de = e - elast
	        dx_norm = Array{Float64}(undef,nroots)
	        conv = [false for i in collect(1:1:nroots)]
	
	        #Check for convergence
	        for (k, ek) in enumerate(e)
	            #Update residual vector
	            xt[:,k] = ax0[:,k] - (ek * x0[:,k])
	            dx_norm[k] = sqrt(dot(xt[:,k], xt[:,k]))
	            conv[k] = (abs(de[k]) < tol) && (dx_norm[k] < toloose)
	            if (conv[k] == true) && (conv_last[k] == false)
	                println("root $k converged")
	            end
	        end
	
	        ax0 = Array{Float64}(undef,nstrings,nroots+trial_space)
	        max_dx_norm = maximum(dx_norm)
	
	        if all(conv)
	            println("Davidson converged in $it iterations")
	            break
	        end
	
	        #Remove subspace linear dependencies
	        if any(((!conv[k]) && (n^2 > lindep)) for (k,n) in enumerate(dx_norm))
	            for (k, ek) in enumerate(e)
	                if (!conv[k]) && (dx_norm[k]^2 > lindep)
	                    xt[:,k] = preconditionner(xt[:,k], e[1])
	                    xt[:,k] *= 1/sqrt(dot(xt[:,k], xt[:,k]))
	                end
	            end
	        else
	            for (k, ek) in enumerate(e)
	                if dx_norm[k]^2 > lindep 
	                    xt[:,k] = preconditionner(xt[:,k],e[1])
	                    xt[:,k] *= 1 / sqrt(dot(xt[:,k], xt[:,k]))
	                else
	                    println("Remove the $k th eigenvector")
	                end
	            end
	        end
	
	        for i in 1:space-1
	            for j in 1:dimt
	                xt[:,j] -= xs[:,i] * dot(xs[:,i], xt[:,j])
	            end
	        end
	        norm_min = 1
	
	        for i in 1:dimt
	            norm = sqrt(dot(xt[:,i], xt[:,i]))
	            if norm^2 > lindep
	                xt[:,i] *= 1 / norm
	                norm_min = minimum([norm_min, norm])
	            else
	                println("Linear dependencies were found in the trial subspace")
	            end
	        end
	
	        if space == 1
	            println("Linear dependencies were found in the trial subspace")
	        end
	    
	        max_dx_last = max_dx_norm
	        fresh_start = (space + nroots) > trial_space
	    end
	    #Check if eigenvector are correctly returned
	    return e, x0[:,1:nroots]
	end
	
	function sort_elast(elast, conv_last, vlast, v, fresh_start)
	    if fresh_start
	        return elast, conv_last
	    end
	    head, nroots = size(vlast)
	    ovlp = broadcast(abs, (v[1:head,:]' * vlast))
	    idx = findmax.(eachrow(ovlp))
	    new_order = [i[2] for i in idx]
	    ordering_diff = (new_order != collect(1:1:length(idx)))
	    if ordering_diff
	        println("Ordering of eigenstates changed : $ordering_diff ")
	    end
	    return [elast[i[2]] for i in idx], [conv_last[i[2]] for i in idx]
	end
	
	function gen_x0(x0, v, xs)
	    space, nroots = size(v)
	    x0[:,1:nroots] = ein"c,x -> xc"(v[space,:],xs[:,space])
	    xsi = Array{Float64}(undef,nstrings, 1)
	    for i in reverse(collect(1:1:(space-1)))
	        xsi  = copy(xs[:,i])
	        for k in 1:nroots
	            x0[:,k] .+= v[i,k] * xsi
	        end
	        xsi = Array{Float64}(undef,nstrings, 1)
	    end
	end
	
	function orthonormalise(xs, dimt, lindep=1e-14)
	    #QR decomposition to form a list of orthonormal vectors
	    nstrings, nvec = size(xs)
	    qs = zeros(nstrings,nvec)
	    nv = 1
	    for i in 1:nvec
	        xi = copy(xs[:,i])
	        for j in 1:nv
	            prod = dot(qs[:,j], xi)
	            xi -= prod * qs[:,j]
	        end
	        innerprod = dot(xi, xi)
	        norm = sqrt(innerprod)
	        if innerprod > lindep
	            qs[:,nv] = xi / norm
	            nv +=1
	        else
	            dimt -= 1
	            println("Warning QR decomposition removed vector n° $i")
	        end
	    end
	    
	    return qs[:,1:(nv-1)], dimt 
	end
	
	function fill_heff!(heff,xs,ax,xt,axt,dimt)
	    #Only for real Hamiltonian
	    nrow = dimt
	    row1 = size(ax)[2]
	    row0 = row1 - nrow
	
	    for (ip,i) in enumerate(range(row0, row1-1))
	        for (jp,j) in enumerate(collect(row0:1:i-1))
	            heff[i+1,j+1] = dot(xt[:,ip], axt[:,jp])
	            heff[j+1,i+1] = heff[i+1,j+1]
	        end
	        heff[(i+1),(i+1)] = dot(xt[:,ip],axt[:,ip])
	    end
	    
	    for i in 0:row0-1
	        for (jp,j) in enumerate(range(row0,row1-1))
	            heff[j+1,i+1] = dot(xt[:,jp], ax[:,i+1])
	            heff[i+1,j+1] = heff[j+1,i+1]
	        end
	    end
	end
end

# ╔═╡ bd7746a8-3601-4e37-800c-267e953ff824
md"
Now we are ready to run our first full-CI calculation
"

# ╔═╡ 0186596c-e43d-4090-9d2a-45e4ec8ca81e
function full_ci(nmo::Int64, nα::Int64 , C::Matrix{Float64}, Hcore::Matrix{Float64}, μνλσ::Array{Float64, 4}, nroots::Int64)
	f1e = zeros(nmo,nmo)
	nstrings = binomial(nmo, nα)
	fcivec = zeros((nstrings * nstrings),nroots)
	j = 1
	
	for i in 1:nroots
		fcivec[j,i] = 1.0
		j += nstrings + i
	end
	
	#AO to MO transformation
	eri = ein"ai,bj,ck,dl,abcd->ijkl"(C,C,C,C,μνλσ)
	
	h1e = C' * Hcore * C
	h2e = absorb_h1e(Hcore, C, eri,nmo,nα*2)
	
	function σ_vec(v)
		σ = contract2e(h2e, v, nmo, nα)
		return vec(σ)
	end
	hdiag = make_hdiag(h1e,eri,nmo,nα)
	precond(x,e)= x ./ (hdiag .- e .+ 1e-4)
	
	e, c = davidson(σ_vec,fcivec,precond)
	
	for i in 1:nroots
		println("E($i) = $(e[i]+V_nuc)")
		println("Full CI done ⌣")
	end
	
	return e, c
end


# ╔═╡ 02e5d8f3-39a6-46ba-88ac-d8ae72d95746
full_ci(nmo, nα, C, Hcore, μνλσ, nroots=2)


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
Combinatorics = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
Fermi = "9237668d-08c8-4784-b8dd-383aa52fcf74"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
OMEinsum = "ebe7aa44-baf0-506c-a96f-8464559b3922"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[compat]
BenchmarkTools = "~1.3.2"
Combinatorics = "~1.0.2"
Fermi = "~0.4.0"
OMEinsum = "~0.7.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.3"
manifest_format = "2.0"
project_hash = "6257c0159aba173c6db4e22f826cacc52b286b67"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "16b6dbc4cf7caee4e1e75c49485ec67b667098a0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.3.1"

[[deps.AbstractTrees]]
git-tree-sha1 = "faa260e4cb5aba097a73fab382dd4b5819d8ec8c"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.4"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "cc37d689f599e8df4f464b2fa3870ff7db7492ef"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.6.1"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArrayInterface]]
deps = ["ArrayInterfaceCore", "Compat", "IfElse", "LinearAlgebra", "SnoopPrecompile", "Static"]
git-tree-sha1 = "dedc16cbdd1d32bead4617d27572f582216ccf23"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "6.0.25"

[[deps.ArrayInterfaceCore]]
deps = ["LinearAlgebra", "SnoopPrecompile", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "e5f08b5689b1aad068e01751889f2f615c7db36d"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.29"

[[deps.ArrayInterfaceOffsetArrays]]
deps = ["ArrayInterface", "OffsetArrays", "Static"]
git-tree-sha1 = "3d1a9a01976971063b3930d1aed1d9c4af0817f8"
uuid = "015c0d05-e682-4f19-8f0a-679ce4c54826"
version = "0.1.7"

[[deps.ArrayInterfaceStaticArrays]]
deps = ["Adapt", "ArrayInterface", "ArrayInterfaceCore", "ArrayInterfaceStaticArraysCore", "LinearAlgebra", "Static", "StaticArrays"]
git-tree-sha1 = "f12dc65aef03d0a49650b20b2fdaf184928fd886"
uuid = "b0d46f97-bff5-4637-a19a-dd75974142cd"
version = "0.1.5"

[[deps.ArrayInterfaceStaticArraysCore]]
deps = ["ArrayInterfaceCore", "LinearAlgebra", "StaticArraysCore"]
git-tree-sha1 = "01a9f8e6cfc2bfdd01d333f70b8014a04893103c"
uuid = "dd5226c6-a4d4-4bc7-8575-46859f9c95b9"
version = "0.1.4"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random", "Test"]
git-tree-sha1 = "a598ecb0d717092b5539dbbe890c98bac842b072"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.2.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BatchedRoutines]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "441db9f0399bcfb4eeb8b891a6b03f7acc5dc731"
uuid = "a9ab73d0-e05c-5df1-8fde-d6a4645b8d8e"
version = "0.2.2"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "d9a9701b899b30332bbcb3e1679c41cce81fb0e8"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.2"

[[deps.BetterExp]]
git-tree-sha1 = "dd3448f3d5b2664db7eceeec5f744535ce6e759b"
uuid = "7cffe744-45fd-4178-b173-cf893948b8b7"
version = "0.1.0"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "0c5f81f47bbbcf4aea7b2959135713459170798b"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.5"

[[deps.Blosc]]
deps = ["Blosc_jll"]
git-tree-sha1 = "310b77648d38c223d947ff3f50f511d08690b8d5"
uuid = "a74b3585-a348-5f62-a45c-50e91977d574"
version = "0.7.3"

[[deps.Blosc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Lz4_jll", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "e94024822c0a5b14989abbdba57820ad5b177b95"
uuid = "0b7ba130-8d10-5ba8-a3d6-c5182647fed9"
version = "1.21.2+0"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "Static"]
git-tree-sha1 = "2c144ddb46b552f72d7eafe7cc2f50746e41ea21"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.2.2"

[[deps.CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CompilerSupportLibraries_jll", "ExprTools", "GPUArrays", "GPUCompiler", "LLVM", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "SpecialFunctions", "TimerOutputs"]
git-tree-sha1 = "6717cb9a3425ebb7b31ca4f832823615d175f64a"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "3.13.1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c6d890a52d2c4d55d326439580c3b8d0875a77d9"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.7"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "485193efd2176b88e6622a39a246f8c5b600e74e"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.6"

[[deps.CloseOpenIntervals]]
deps = ["ArrayInterface", "Static"]
git-tree-sha1 = "d61300b9895f129f4bd684b2aff97cf319b6c493"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.11"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSolve]]
git-tree-sha1 = "9441451ee712d1aec22edad62db1a9af3dc8d852"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.3"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "550e3f8a429fd2f8a9da59f1589c5e268ddc97b3"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.46.1"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "89a9db8d28102b094992472d333674bd1a83ce2a"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.1"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "e8119c1a33d267e16108be441a287a6981ba1630"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.14.0"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "a4ad7ef19d2cdc2eff57abbbe68032b1cd0bd8f8"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.13.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.ExprTools]]
git-tree-sha1 = "c1d06d129da9f55715c6c212866f5b1bddc5fa00"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.9"

[[deps.Fermi]]
deps = ["Formatting", "GaussianBasis", "LinearAlgebra", "LoopVectorization", "Memoize", "Molecules", "Octavian", "PrettyTables", "Strided", "Suppressor", "TBLIS", "TensorOperations", "Test", "libcint_jll"]
git-tree-sha1 = "8cb6707d5aca1eed3d6194a2f154ddccaa47e246"
uuid = "9237668d-08c8-4784-b8dd-383aa52fcf74"
version = "0.4.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "00e252f4d706b3d55a8863432e742bf5717b498d"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.35"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
git-tree-sha1 = "28231bc5bee6672747be0313dc349c08f42942f6"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "8.6.4"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "1cd7f0af1aa58abc02ea1d872953a97359cb87fa"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.4"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "19d693666a304e8c371798f4900f7435558c7cde"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "0.17.3"

[[deps.GaussianBasis]]
deps = ["Formatting", "HDF5", "Molecules", "Test", "libcint_jll"]
git-tree-sha1 = "b99d9030424a2199021208e8ba781e6b8f774a3d"
uuid = "9bb1a3dc-0d1c-467e-84f5-0c4ef701360a"
version = "0.2.1"

[[deps.HDF5]]
deps = ["Blosc", "Compat", "HDF5_jll", "Libdl", "Mmap", "Random", "Requires"]
git-tree-sha1 = "698c099c6613d7b7f151832868728f426abe698b"
uuid = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
version = "0.15.7"

[[deps.HDF5_jll]]
deps = ["Artifacts", "JLLWrappers", "LibCURL_jll", "Libdl", "OpenSSL_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "4cc2bb72df6ff40b055295fdef6d92955f9dede8"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.12.2+2"

[[deps.HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Static"]
git-tree-sha1 = "734fd90dd2f920a2f1921d5388dcebe805b262dc"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.14"

[[deps.Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a35518b15f2e63b60c44ee72be5e3a8dbf570e1b"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.9.0+0"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "49510dfcb407e572524ba94aeae2fced1f3feb0f"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.8"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Printf", "Unicode"]
git-tree-sha1 = "df115c31f5c163697eede495918d8e85045c8f04"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "4.16.0"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg", "TOML"]
git-tree-sha1 = "7718cf44439c676bc0ec66a87099f41015a522d6"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.16+2"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f689897ccbe049adb19a065c495e75f372ecd42b"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.4+0"

[[deps.LRUCache]]
git-tree-sha1 = "d862633ef6097461037a00a13f709a62ae4bdfdd"
uuid = "8ac3fa9e-de4c-5943-b1dc-09c6b5f20637"
version = "1.4.0"

[[deps.LayoutPointers]]
deps = ["ArrayInterface", "ArrayInterfaceOffsetArrays", "ArrayInterfaceStaticArrays", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static"]
git-tree-sha1 = "0ad6f0c51ce004dadc24a28a0dfecfb568e52242"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.13"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "0a1b7c2863e44523180fdb3146534e265a91870b"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.23"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoopVectorization]]
deps = ["ArrayInterface", "ArrayInterfaceCore", "ArrayInterfaceOffsetArrays", "ArrayInterfaceStaticArrays", "CPUSummary", "ChainRulesCore", "CloseOpenIntervals", "DocStringExtensions", "ForwardDiff", "HostCPUFeatures", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "PolyesterWeave", "SIMDTypes", "SLEEFPirates", "SnoopPrecompile", "SpecialFunctions", "Static", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "9696a80c21a56b937e3fd89e972f8db5db3186e2"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.150"

[[deps.Lz4_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "5d494bc6e85c4c9b626ee0cab05daa4085486ab1"
uuid = "5ced341a-0733-55b8-9ab6-a4889d929147"
version = "1.9.3+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Measurements]]
deps = ["Calculus", "LinearAlgebra", "Printf", "RecipesBase", "Requires"]
git-tree-sha1 = "12950d646ce04fb2e89ba5bd890205882c3592d7"
uuid = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
version = "2.8.0"

[[deps.Memoize]]
deps = ["MacroTools"]
git-tree-sha1 = "2b1dfcba103de714d31c033b5dacc2e4a12c7caa"
uuid = "c03570c3-d221-55d1-a50c-7939bbd78826"
version = "0.4.4"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.Molecules]]
deps = ["Formatting", "LinearAlgebra", "PeriodicTable", "PhysicalConstants", "Test", "Unitful"]
git-tree-sha1 = "01f3bb4350e3a5cfaef68bab9b9e4df6cbc9d980"
uuid = "5de6a177-b489-40a9-b2f4-524242b9b679"
version = "0.1.2"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OMEinsum]]
deps = ["AbstractTrees", "BatchedRoutines", "CUDA", "ChainRulesCore", "Combinatorics", "LinearAlgebra", "MacroTools", "OMEinsumContractionOrders", "Requires", "Test", "TupleTools"]
git-tree-sha1 = "8d86af7f948ab3f24a67b833c3cea5e0915b5842"
uuid = "ebe7aa44-baf0-506c-a96f-8464559b3922"
version = "0.7.2"

[[deps.OMEinsumContractionOrders]]
deps = ["AbstractTrees", "BetterExp", "JSON", "Requires", "SparseArrays", "Suppressor"]
git-tree-sha1 = "0d4fbd4f2d368bf104671187dcd716a2fac533e0"
uuid = "6f22d1fd-8eed-4bb7-9776-e7d684900715"
version = "0.8.1"

[[deps.Octavian]]
deps = ["ArrayInterface", "CPUSummary", "IfElse", "LoopVectorization", "ManualMemory", "PolyesterWeave", "Requires", "SnoopPrecompile", "Static", "ThreadingUtilities", "VectorizationBase"]
git-tree-sha1 = "b6c8c7f574c981546cf7b0f017572e5d019991a3"
uuid = "6fd5a793-0b7e-452c-907f-f8bfe9c57db4"
version = "0.3.20"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "82d7c9e310fe55aa54996e6f7f94674e2a38fcb4"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.9"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9ff31d101d987eb9d66bd8b176ac7c277beccd09"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.20+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.Parsers]]
deps = ["Dates", "SnoopPrecompile"]
git-tree-sha1 = "478ac6c952fddd4399e71d4779797c538d0ff2bf"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.8"

[[deps.PeriodicTable]]
deps = ["Base64", "Test", "Unitful"]
git-tree-sha1 = "5ed1e2691eb13b6e955aff1b7eec0b2401df208c"
uuid = "7b2266bf-644c-5ea3-82d8-af4bbd25a884"
version = "1.1.3"

[[deps.PhysicalConstants]]
deps = ["Measurements", "Roots", "Unitful"]
git-tree-sha1 = "cd4da9d1890bc2204b08fe95ebafa55e9366ae4e"
uuid = "5ad8b20f-a522-5ce9-bfc9-ddf1d5bda6ab"
version = "0.2.3"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "240d7170f5ffdb285f9427b92333c3463bf65bf6"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "7a1a306b72cfa60634f03a911405f4e64d1b718b"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.6.0"

[[deps.RandomNumbers]]
deps = ["Random", "Requires"]
git-tree-sha1 = "043da614cc7e95c703498a491e2c21f58a2b8111"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.5.3"

[[deps.RecipesBase]]
deps = ["SnoopPrecompile"]
git-tree-sha1 = "261dddd3b862bd2c940cf6ca4d1c8fe593e457c8"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.3"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Roots]]
deps = ["ChainRulesCore", "CommonSolve", "Printf", "Setfield"]
git-tree-sha1 = "b45deea4566988994ebb8fb80aa438a295995a6e"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "2.0.10"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "cda0aece8080e992f6370491b08ef3909d1c04e7"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.38"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "ef28127915f4229c971eb43f3fc075dd3fe91880"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.2.0"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "08be5ee09a7632c32695d954a602df96a877bf0d"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.8.6"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "6aa098ef1012364f2ede6b17bf358c7f1fbe90d4"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.17"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.Strided]]
deps = ["LinearAlgebra", "TupleTools"]
git-tree-sha1 = "a7a664c91104329c88222aa20264e1a05b6ad138"
uuid = "5e0ebb24-38b0-5f93-81fe-25c709ecae67"
version = "1.2.3"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.Suppressor]]
git-tree-sha1 = "c6ed566db2fe3931292865b966d6d140b7ef32a9"
uuid = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
version = "0.2.1"

[[deps.TBLIS]]
deps = ["Hwloc_jll", "Libdl", "LinearAlgebra", "Test", "tblis_jll"]
git-tree-sha1 = "4d27757bbe946c88d896a57d000f41c36dd3248e"
uuid = "48530278-0828-4a49-9772-0f3830dfa1e9"
version = "0.2.0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "1544b926975372da01227b382066ab70e574a3ec"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

[[deps.TensorOperations]]
deps = ["CUDA", "LRUCache", "LinearAlgebra", "Requires", "Strided", "TupleTools"]
git-tree-sha1 = "c082dda2ace9de2bc71b644ae29e2adf1a8137b2"
uuid = "6aa20fa7-93e2-5fca-9bc0-fbd0db3c71a2"
version = "3.2.4"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "c97f60dd4f2331e1a495527f80d242501d2f9865"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.1"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "f2fd3f288dfc6f507b0c3a2eb3bac009251e548b"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.22"

[[deps.TupleTools]]
git-tree-sha1 = "3c712976c47707ff893cf6ba4354aa14db1d8938"
uuid = "9d95972d-f1c8-5527-a6e0-b4b365fa01f6"
version = "1.3.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Unitful]]
deps = ["ConstructionBase", "Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "bb37ed24f338bc59b83e3fc9f32dd388e5396c53"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.12.4"

[[deps.VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static"]
git-tree-sha1 = "4c59c2df8d2676c4691a39fa70495a6db0c5d290"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.58"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c6edfe154ad7b313c01aceca188c05c835c67360"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.4+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.libcint_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6b98084051c2a3d073268d0349108847361a2b3f"
uuid = "574b78ca-bebd-517c-801d-4735c93a9686"
version = "5.1.7+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.tblis_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl", "Pkg"]
git-tree-sha1 = "c1ef28e5eaddc60824960abbc1190d7d5046f261"
uuid = "9c7f617c-f299-5d18-afb6-044c7798b3d0"
version = "1.2.0+4"
"""

# ╔═╡ Cell order:
# ╠═d050a7f8-c4c1-11ed-3542-7dffdbf79b0f
# ╟─2f904006-4f00-4046-b78d-e9792acb5b96
# ╠═d415f46c-b803-4f69-9290-b7e722606660
# ╟─c6c9efc5-2ee4-4de1-8018-ac90d167c655
# ╟─f6d71eef-2450-4612-81d5-43fb46c6fb1b
# ╠═f997e31f-3844-4f84-b014-d211ab4f5997
# ╟─6c5fdf02-d0f3-4471-8e27-30ee26036bd1
# ╟─e94cc0cc-016a-4cbd-9dfe-b98d9f012556
# ╠═8afd6650-e4ca-4a58-8aec-ed31d8dcdb7f
# ╟─9149e499-2d7f-4d53-a7a3-8495cf8c3aa1
# ╠═0a6f1c6c-f4c0-4ce0-8848-240eaab58839
# ╟─9b28dc61-6da2-4230-aa26-aa57ddd97315
# ╠═afd13da3-8825-41c5-a897-41234fdd89ea
# ╟─b427df39-c75f-41a4-b001-04311d1e62fb
# ╠═b7351384-f2bb-4a29-95aa-0e588c9b48e8
# ╟─06f727d6-092b-4256-9b3d-478fbcbf5671
# ╠═44d0daad-87f7-4c09-a529-78ec4bbebf0b
# ╟─c2920c78-3393-47d5-a9d8-100e7e0cd7ef
# ╠═3ea77dfa-6f24-4f7b-92f6-d02e22aebe1f
# ╟─8a57eead-590d-4eab-b27e-5be826de3005
# ╠═fe2b36e9-3a68-4a0c-9549-79e0a0fddb73
# ╟─e5fef9ac-4cd0-47a7-a3c6-e44e3be71cdd
# ╠═a3753425-2ca4-4759-ad9c-cdb14571907f
# ╟─ebf029cf-b7af-47ac-9b48-ca303e136f48
# ╠═1c4f0b30-4dff-4f06-83e2-93307f3769c8
# ╟─bd7746a8-3601-4e37-800c-267e953ff824
# ╠═0186596c-e43d-4090-9d2a-45e4ec8ca81e
# ╠═02e5d8f3-39a6-46ba-88ac-d8ae72d95746
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
