#= 
    Initialize Package:

    Lux : a lightweight, flexible deep learning library for Julia
    Random:
    Optimisers: library for optimization algorithms (like Adam, SGD) that update model weights during training.
    Zygote: 


    STEP 1: Define the NN
    STEP 2: Define the ODE problem
    STEP 3: Solve the ODE
    STEP 4: Define a loss function for training
    STEP 5: Training loop
    STEP 6: Inspect or plot results

=# 
using Lux, Random, Optimisers, Zygote

# STEP 1

dudt = Chain(
    Dense(2 => 50, relu), # input 2 -> 50 hidden units, Rectified Linear Unit activation -> relu(x) = max(0,x)
    Dense(50 => 2) # hidden 50 -> output 2, linear activation
)
ps = params(dudt) # extract trainable parameters

# STEP 2

# Function defining how u (state) changes over time (t): 
# u = vector of current state variables, (all the variables of the system at time t)
# p = parameters (unused because dudt already captures them), t = time
# dudt(u) -> NN predicts du/dt (derivative of u)
function f(u, p, t)
    dudt(u) # NN predicts derivative du/dt
end

u0 = [1.0, 0.0] # initial condition
tspan = (0.0, 1.0) # start time -> end time (interval over which to solve the ODE)

# STEP 3

# ODEProblem -> constructs a problem object for the solver
prob = ODEProblem(f, u0, tspan)

sol = solve(prob) # uses OrdinaryDiffEqDefault automatically // Numerically integrates the ODE over tspan

# STEP 4


# A measure of how far the network's prediction is from the true solution
function loss()
    sol = solve(prob)
    predicted = sol.u[end]                # final state predicted by NN
    target = [0.0, 1.0]                   # desired final state 
    return sum((predicted .- target).^2)  # mean squared error
end

# STEP 5

opt = Optimisers.Adam(0.01) # learning rate 0.01 using Adam Optimiser

# learning rate -> step size when updating weights
# epoch -> one pass through the training procedure (we're going through it 100 times)
# gradient -> direction to change weights to reduce error

for epoch in 1:100
    grads = gradient(() -> loss(), ps) # computes derivative of loss w.r.t. network parameters
    Optimisers.update!(opt, ps, grads) # update network parameters / modifies weights to reduce loss
end

# STEP 6

plot(sol.t, hcat(sol.u...)', label=["x1" "x2"])

# hcat(sol.u...)' converts array of vectors into a matrix for plotting
