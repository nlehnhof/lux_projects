using Lux,
    Optimization,
    OptimizationOptimisers,
    OptimizationOptimJL,
    OrdinaryDiffEqTsit5,
    SciMLSensitivity,
    Random,
    MLUtils,
    CairoMakie,
    ComponentArrays,
    Printf

const gdev = gpu_device()
const cdev = cpu_device()

function lotka_volterra(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = α * x - β * x * y
    du[2] = -δ * y + γ * x * y
    return nothing
end

u0 = [1.0f0, 1.0f0]

datasize = 32
tspan = (0.0f0, 2.0f0)

const t = range(tspan[1], tspan[2]; length=datasize)
true_prob = ODEProblem(lotka_volterra, u0, (tspan[1], tspan[2]), [1.5, 1.0, 3.0, 1.0])
const ode_data = Array(solve(true_prob, Tsit5(); saveat=t))

begin
    fig = Figure()
    ax = CairoMakie.Axis(fig[1, 1])
    lines!(ax, t, ode_data[1, :]; label=L"u_1(t)", color=:blue, linestyle=:dot, linewidth=4)
    lines!(ax, t, ode_data[2, :]; label=L"u_2(t)", color=:red, linestyle=:dot, linewidth=4)
    axislegend(ax; position=:lt)
    fig
end

