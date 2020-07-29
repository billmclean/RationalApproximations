import RationalApproximations: aaa_v1, aaa_v2, locate_extrema_idx
import RationalApproximations: RationalFunc, equioscillation!
using PyPlot

f = exp
S = range(-1, 1, length=201)
F = f.(S)
tol = 1e-6
max_m = 5
r, err = aaa_v1(F, S, max_m, tol)
m = length(r)

figure(1)
supp_pt = r[m].supp_pt
resid = F - r[m].(S)
plot(S, resid, supp_pt, f.(supp_pt)-r[m].(supp_pt), "o")
legend((L"$f-r$", "support points"))
title("Result using aaa_v1 (m = $m)")
grid(true)
xylims = axis()

figure(2)
r, err = aaa_v2(F, S, max_m, tol)
m = length(r)
supp_pt = r[m].supp_pt
resid = F - r[m].(S)
plot(S, resid, supp_pt, f.(supp_pt)-r[m].(supp_pt), "o")
legend((L"$f-r$", "support points"))
title("Result using aaa_v2 (m = $m)")
grid(true)
axis(xylims)

idx = locate_extrema_idx(resid, m)
x = S[idx]
figure(3)
plot(S, resid, x, f.(x)-r[m].(x), "o")
legend((L"$f-r$", "extrema"))
title("Locate extrema")
grid(true)
axis(xylims)

α, β, λ = equioscillation!(f, x, supp_pt)
new_r = RationalFunc(α, β, supp_pt)
figure(4)
plot(S, f.(S) - new_r.(S),
     [-1.0, 1.0, NaN, -1.0, 1.0], [λ, λ, NaN, -λ, -λ], ":k",
     x, f.(x)-new_r.(x), "o")
grid(true)
title("Equioscillation procedure")
axis(xylims)

max_iterations = 4
x = S[idx]
r, zmin, zmax = minimax(f, x, supp_pt, (-1.0,1.0), max_iterations)
m = length(r)
figure(5)
plot(S, f.(S) - r[m].(S), x, f.(x) - r[m].(x), "o")
grid(true)
title("Result after $m Remez iterations")
axis(xylims)

