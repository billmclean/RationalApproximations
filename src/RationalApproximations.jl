module RationalApproximations

import GenericSVD: svd!
import LinearAlgebra: norm

export aaa_v1, aaa_v2, minimax

"""
Represents a rational function of type `(m,m)` in barycentric form.
"""
struct RationalFunc{T<:AbstractFloat}
    α::Vector{T}
    β::Vector{T}
    supp_pt::Vector{T}
end # module

function (r::RationalFunc{T})(x::T) where T <: AbstractFloat
    numer = zero(T)
    denom = zero(T)
    for k = 1:length(r.α)
        δ = x - r.supp_pt[k]
        if abs(δ) ≤ eps(x)
            numer = r.α[k]
            denom = r.β[k]
            break
        else
            numer += r.α[k] / δ
            denom += r.β[k] / δ
        end
    end
    return numer / denom
end

"""
    aaa_v1(F, S, max_m, tol) -> r, err

Adaptive Antoulas-Anderson algorithm computes rational approximations `r[m]`of
type `(m,m)` to a real function `f(x)` based on sample values `F[i] = f(S[i])`.
Stops if `err[m] < tol * norm(F, Inf)` or when `m` reaches `max_m`.

This is version 1 of the algorithm, for which `r[m]` interpolates `f` as the
support points based on `m` weights `w[m]` with `α[j] = w[j] * F[j]` and
`β[j] = w[j]`.
"""
function aaa_v1(F::Vector{T}, S::AbstractVector{T},
                    max_m::Integer, tol::T) where T <: AbstractFloat
    K = length(F)
    if length(S) ≠ K
        throw(DimensionMismatch("F and S must have the same length"))
    end
    avg_F = sum(F) / K
    R = fill(avg_F, K)
    J = collect(1:K)
    supp_pt = T[]
    f = T[]
    w = T[]
    err = T[]
    r = RationalFunc[]
    for m = 1:max_m
        val, i = findmax(abs.(F[J]-R[J]))
        j = J[i]
        push!(supp_pt, S[j])
        push!(f, F[j])
        J = remove(J, i)
        A = T[ (F[J[i]]-f[j])/(S[J[i]]-supp_pt[j]) for i=1:K-m, j=1:m ]
        Fact = svd!(A)
        w = Fact.V[:,m]
        α = w .* f
        β = w
        push!(r, RationalFunc(α, β, copy(supp_pt)))
        R[J] .= r[m].(S[J])
        push!(err, norm(F[J] - R[J], Inf))
        if err[m] < tol * norm(F, Inf)
            r = r[1:m]
            err = err[1:m]
            break
        end
    end
    return r, err
end

"""
    remove(J, i)

Returns the vector obtained from `J` by removing the `i`th element.
"""
function remove(J::Vector, i::Integer)
    m = length(J)
    if i ≤ 0 || i ≥ m+1
        throw(BoundsError(J, i))
    end
    return [J[1:i-1]; J[i+1:m]]
end

"""
    aaa_v1(F, S, max_m, tol) -> r, err

Adaptive Antoulas-Anderson algorithm computes rational approximations `r[m]`of
type `(m,m)` to a real function `f(x)` based on sample values `F[i] = f(S[i])`.
Stops if `err[m] < tol * norm(F, Inf)` or when `m` reaches `max_m`.

This is version 2 of the algorithm, for which the coefficients `α[j]` and `β[j]`
vary independently.
"""
function aaa_v2(F::Vector{T}, S::AbstractVector{T},
                    max_m::Integer, tol::T) where T <: AbstractFloat
    K = length(F)
    if length(S) ≠ K
        throw(DimensionMismatch("F and S must have the same length"))
    end
    avg_F = sum(F) / K
    R = fill(avg_F, K)
    J = collect(1:K)
    supp_pt = T[]
    supp_pt_idx = fill(-1, K)
    resid = F - R
    f = T[]
    w = T[]
    err = T[]
    r = RationalFunc[]
    buffer = Vector{T}(undef, K*2*max_m)
    for m = 1:max_m
        val, l = findmax(abs.(resid[1:K-m+1]))
        i = 0
        for i1 = 1:l
            i += 1
            if supp_pt_idx[i] ≥ 0
                i += 1
            end
        end
        supp_pt_idx[i] = m
        push!(supp_pt, S[i])
        push!(f, F[i])
        A = reshape(view(buffer, 1:(K-m)*2*m), K-m, 2m)
        for j = 1:m
            i1 = 0
            for i = 1:K
                if supp_pt_idx[i] ≤ 0
                    i1 += 1
                    A[i1,j]   = -1 / ( S[i] - supp_pt[j] )
                    A[i1,m+j] =  F[i] / ( S[i] - supp_pt[j] )
                end
            end
        end
        Fact = svd!(A)
        α = Fact.V[1:m,2m]
        β = Fact.V[m+1:2m,2m]
        push!(r, RationalFunc(α, β, copy(supp_pt)))
        i1 = 0
        for i = 1:K
            if supp_pt_idx[i] ≤ 0
                i1 += 1
                resid[i1] = F[i] - r[m](S[i])
            end
        end
        push!(err, norm(resid[1:K-m], Inf))
        if err[m] ≤ tol * norm(F, Inf)
            break
        end
    end
    return r, err
end

"""
    minimax(f, x, supp_pt, Interval, max_iterations) -> r, zmin, zmax
"""
function minimax(f::Function, x::Vector{T}, supp_pt::Vector{T},
                 Interval::Tuple{T,T}, 
                 max_iterations::Integer) where T <: AbstractFloat
    N = length(supp_pt)
    if length(x) ≠ 2N
        throw(DimensionMismatch("x must be twice as long as supp_pt"))
    end
    α = Vector{T}(undef, N)
    β = similar(α)
    δ = one(T)
    zmax = Vector{T}(undef, max_iterations)
    zmin = similar(zmax)
    r = Vector{RationalFunc{T}}(undef, max_iterations)
    for i = 1:max_iterations
        α, β, λ = equioscillation!(f, x, supp_pt)
        r[i] = RationalFunc(α, β, supp_pt)
        zmin[i], zmax[i] = locate_extrema!(x, Interval) do t
            return f(t) - r[i](t)
        end
        δ = zmax[i] - zmin[i]
        if δ ≤ 10*eps(zmax[i])
            zmax = zmax[1:i]
            zmin = zmin[1:i]
            r = r[1:i]
            break
        end
    end
    return r, zmin, zmax
end

"""
    locate_extrema_idx(resid, m)
"""
function locate_extrema_idx(resid::Vector{T}, 
                            m::Integer) where T <: AbstractFloat
    K = length(resid)
    idx = Int64[]
    push!(idx, 1)
    fwd = resid[2] - resid[1]
    σ = sign(fwd)
    for k = 2:K-1
        fwd = resid[k+1] - resid[k]
        if σ * sign(fwd) == -1.0
            push!(idx, k)
            σ = -σ
        end
    end
    push!(idx, K)
    len_idx = length(idx)
    if len_idx < 2m
        error("Found only $len_idx extrema; need $(2m)")
    elseif len_idx > 2m
        var = abs.(diff(resid[idx]))
        I = sortperm(var, rev=true)
        new_idx = [ idx[I[1:2m-1]]; idx[end] ]
        idx = sort(new_idx)
    end
    return idx
end

"""
    equioscillation!(f, x, supp_pt) -> α, β, λ
"""
function equioscillation!(f::Function, x::Vector{T},
                         supp_pt::Vector{T}) where T <: AbstractFloat
    N = length(supp_pt)
    if length(x) ≠ 2N
        throw(DimensionMismatch("x must be twice as long as supp_pt"))
    end
    M = Matrix{T}(undef, 2N, N)
    for k = 1:N
        for l = 1:2N
            Mlk = one(T)
            for p = 1:k-1
                Mlk *= x[l] - supp_pt[p]
            end
            for p = k+1:N
                Mlk *= x[l] - supp_pt[p]
            end
            M[l,k] = Mlk
        end
    end
    d = Vector{Float64}(undef, 2N)
    for l = 1:2N
        Dl = one(T)
        for p = 1:l-1
            Dl *= x[l] - x[p]
        end
        for p = l+1:2N
            Dl *= x[l] - x[p]
        end
        d[l] = 1/sqrt(abs(Dl))
    end
    D = Diagonal(d)
    Fact = qr(D*M)
    Q1, R = Matrix(Fact.Q), Fact.R
    sgn = Vector{Float64}(undef, 2N)
    sgn[1] = -one(T)
    for l = 2:2N
        sgn[l] = -sgn[l-1]
    end
    S = Diagonal(sgn)
    F = Diagonal(f.(x))
    A = Q1' * ( S * F ) * Q1
    EFact = eigen(A)
    y = EFact.vectors
    Q1y = Q1 * y
    q = D \ Q1y
    k_no_root = -1
    check_count = 0
    for k = 1:N
        no_root = true
        for l = 2:2N
            if sign(q[l,k]) != sign(q[1,k])
                no_root = false
            end
        end
        if no_root
            k_no_root = k
            check_count += 1
        end
    end
    if check_count ≠ 1
        error("Failed to find unique eigenpair with q≠0")
    end
    α = R \ ( Q1' * F * Q1y[:,k_no_root] )
    β = R \ y[:,k_no_root]
    λ = EFact.values[k_no_root]
    return α, β, λ
end

"""
    locate_extrema!(resid, x, Interval) -> zmin, zmax
"""
function locate_extrema!(resid::Function, x::Vector{Float64},
                         Interval::Tuple{Float64,Float64})
    n = length(x)
    if resid(x[1]) < resid(x[2])
        sgn = 1
    else
        sgn = -1
    end
    res = optimize(Interval[1], x[2]) do t
        return sgn * resid(t)
    end
    x[1] = res.minimizer
    zmax = zmin = abs(resid(x[1]))
    for l = 2:n-1
        sgn = -sgn
        res = optimize(x[l-1], x[l+1]) do t
            return sgn * resid(t)
        end
        x[l] = res.minimizer
        zmax = max(zmax, abs(resid(x[l])))
        zmin = min(zmin, abs(resid(x[l])))
    end
    sgn = -sgn
    res = optimize(x[n-1], Interval[2]) do t
        return sgn * resid(t)
    end
    x[n] = res.minimizer
    zmax = max(zmax, abs(resid(x[n])))
    zmin = min(zmin, abs(resid(x[n])))
    return zmin, zmax
end

end # module
