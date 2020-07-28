using RationalApproximations
using PyPlot

S = range(-1, 1, length=100)
F = exp.(S)
max_m = 5
tol = 1e-6
r_v1, err = aaa_v1(F, S, max_m, tol)

figure(1)
t = range(-1, 1, length=201)
y = exp.(t)
for m = 2:4
    semilogy(t, abs.(y-r_v1[m].(t)), label="m=$m")
end
legend(loc="lower left")
grid(true)
title("AAA Version 1: absolute errors")

r_v2, err = aaa_v2(F, S, max_m, tol)

figure(2)
for m = 2:4
    semilogy(t, abs.(y-r_v2[m].(t)), label="m=$m")
end
legend(loc="lower left")
grid(true)
title("AAA Version 2: absolute errors")

figure(3)
plot(t, y-r_v1[4].(t), label="V1")
plot(t, y-r_v2[4].(t), label="V2")
legend()
grid(true)
title("Comparison of errors when m=4")
