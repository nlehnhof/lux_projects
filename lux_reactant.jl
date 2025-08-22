using Lux, Reactant
# using Lux, Random, Optimisers, Reactant, Enzyme
# using Printf # For pretty printing

# dev = reactant_device()

# n_in = 1
# n_out = 1
# nlayers = 3

# model = @compact(
#     w1=Dense(n_in => 32),
#     w2=[Dense(32 => 32) for i in 1:nlayers],
#     w3=Dense(32 => n_out),
#     act=relu
# ) do x
#     embed = act(w1(x))
#     for w in w2
#         embed = act(w(embed))
#     end
#     out = w3(embed)
#     @return out
# end

# ### INITIALIZE AND TRAIN ###

# rng = Random.default_rng()
# Random.seed!(rng, 0)

# ps, st = Lux.setup(rng, model) |> dev

# x = rand(rng, Float32, n_in, 32) |> dev

# @jit model(x, ps, st)  # 1×32 Matrix and updated state as output.

# x_data = reshape(collect(-2.0f0:0.1f0:2.0f0), 1, :)
# y_data = 2 .* x_data .- x_data .^ 3
# x_data, y_data = dev(x_data), dev(y_data)

# function train_model!(model, ps, st, x_data, y_data, num_epochs=1000)
#     train_state = Lux.Training.TrainState(model, ps, st, Adam(0.001f0))

#     for iter in 1:num_epochs
#         _, loss, _, train_state = Lux.Training.single_train_step!(
#             AutoEnzyme(), MSELoss(),
#             (x_data, y_data), train_state
#         )
#         if iter == 1 || iter % 100 == 0 || iter == num_epochs
#             @printf "Iteration: %04d \t Loss: %10.9g\n" iter loss
#         end
#     end

#     return model, ps, st
# end

# train_model!(model, ps, st, x_data, y_data)