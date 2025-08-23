using CSV, DataFrames, Random, Statistics, Lux, Optimisers, Zygote
using MLUtils
using Plots, Printf

function main()
    # -------------------------
    # Load CSV Data
    # -------------------------
    csv_path = joinpath(homedir(), "Downloads", "auto-mpg.csv")
    df = CSV.read(csv_path, DataFrame; missingstring="?")
    dropmissing!(df)

    numeric_cols = [:mpg, :cylinders, :displacement, :horsepower,
                    :weight, :acceleration, :model_year, :origin]
    foreach(c -> df[!, c] = Float64.(df[!, c]), numeric_cols)

    # Features & labels
    X = Matrix(select(df, Not([:mpg, :car_name])))
    y = Float64.(df.mpg)

    # Standardize features
    X_mean = mean(X, dims=1)
    X_std  = std(X, dims=1)
    X_std[X_std .== 0] .= 1.0
    Xz = (X .- X_mean) ./ X_std

    y_mean = mean(y)
    y_std = std(y)
    y_z = (y .- y_mean) ./ y_std

    # -------------------------
    # Train/Test Split
    # -------------------------
    rng = MersenneTwister(0)
    n_samples = size(Xz, 1)
    idx = shuffle(rng, 1:n_samples)
    train_n = Int(floor(0.8 * n_samples))
    train_idx, test_idx = idx[1:train_n], idx[train_n+1:end]

    Xtr, ytr = Float32.(Xz[train_idx, :]), Float32.(y_z[train_idx])
    Xte, yte = Float32.(Xz[test_idx, :]), Float32.(y_z[test_idx])

    train_loader = MLUtils.DataLoader((Xtr', reshape(ytr,1,:)); batchsize=16, shuffle=false)

    # -------------------------
    # Define a simpler model
    # -------------------------
    model = Chain(
        Dense(size(Xtr,2) => 128, relu),
        Dense(128 => 64, relu),
        Dense(64 => 32, relu),
        Dense(32 => 1)
    )

    ps, st = Lux.setup(rng, model) # Initialize parameters and states

    opt = Optimisers.AdamW(0.01, (0.99, 0.999), 0, 1e-8)  # slightly higher learning rate for faster convergence
    train_state = Lux.Training.TrainState(model, ps, st, opt)

    # -------------------------
    # Training step
    # -------------------------
    function train_step!(state, xb, yb)
        gs, loss, stats, state = Training.single_train_step!(
            AutoZygote(), MSELoss(), (xb, yb), state
        )
        return loss, state
    end

    # -------------------------
    # Training loop
    # -------------------------
    
    nepoch = range(240, 400, 20)
    num_epochs = 282
    train_loss = []
    test_loss = []

    for each in eachindex(nepoch)
        num_epochs = each
        train_losses = Float32[]

        for epoch in 1:num_epochs
            epoch_loss = 0.0
            for (xb, yb) in train_loader
                loss, train_state = train_step!(train_state, xb, yb)
                epoch_loss += loss * size(xb,2)
            end
            epoch_loss /= size(Xtr,1)
            push!(train_losses, epoch_loss)
            if epoch % 10 == 0 || epoch == num_epochs
                @printf("Epoch %3d | Train Loss: %.4f\n", epoch, epoch_loss)
            end
            if epoch == num_epochs
                push!(train_loss, epoch_loss)
            end

            # -------------------------
            # Evaluate on test set
            # -------------------------
        end
            y_pred_z, _ = Lux.apply(model, Xte', ps, st)
            y_pred = vec(y_pred_z) .* y_std .+ y_mean   # de-standardize
            y_actual = vec(yte) .* y_std .+ y_mean

            test_mse = mean((y_pred .- y_actual).^2)
            push!(test_loss, test_mse)
            println("\nFinal Test MSE: ", test_mse)
    
            # scatter(y_actual, y_pred, xlabel="Actual MPG", ylabel="Predicted MPG",
            # title="Actual vs Predicted MPG", color=:dodgerblue, label = "NN Predictions")
            # plot!(y_actual, y_actual, linestyle=:dash, color=:red, label = "Perfect Fit")  # perfect line
    end

    # -------------------------
    # Plot predictions for ONE
    # -------------------------

    # --- PLOT LOSSES FOR DIFFERENT EPOCH NUMBERS ---
    plot(nepoch, train_loss, label = "Train Loss")
    plot!(nepoch, test_loss, label = "Test Loss")
    xlabel!("Epochs")
    ylabel!("Loss")
end

main()
