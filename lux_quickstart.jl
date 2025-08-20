using Lux, Random, Optimisers, Zygote

# -----------------------
# 1️⃣ Setup
# -----------------------
rng = Random.default_rng()
Random.seed!(rng, 0)

# Model: 128 → 256 → 1 → 10
model = Chain(
    Dense(128, 256, tanh),
    Dense(256, 1, tanh),
    Dense(1, 10)
)

# Device
dev = gpu_device()

# Parameters and state
ps, st = Lux.setup(rng, model) |> dev

# Dummy input and target
x = rand(rng, Float32, 128, 2) |> dev          # 128 features × 2 samples
y_true = rand(rng, Float32, 10, 2) |> dev      # 10 outputs × 2 samples

# Initial forward pass
y, st = Lux.apply(model, x, ps, st)

# Training state with Adam
train_state = Lux.Training.TrainState(model, ps, st, Adam(0.0001f0))

# -----------------------
# 2️⃣ Training function
# -----------------------
function train_step!(train_state, x, y)
    gs, loss, stats, train_state = Training.single_train_step!(
        AutoZygote(), MSELoss(), (x, y), train_state
    )
    return gs, loss, stats, train_state
end

# -----------------------
# 3️⃣ Training loop
# -----------------------
num_steps = 10000

for step in 1:num_steps
    global train_state
    gs, loss, stats, train_state = train_step!(train_state, x, y_true)
    if step % 1000 == 0
        println("Step $step, Loss = $loss")
    end
end
