using LinearAlgebra
using Statistics
using Random
using Distributions
using Flux
using MLUtils
using MLDatasets
using Flux: onehotbatch, onecold, DataLoader, Optimiser
using BSON:@save,@load
using ProgressMeter
using OneHotArrays
using Parameters



function block(in_channels::Int, kernel_size::Int)
    layer = SkipConnection(
        Chain(DepthwiseConv((kernel_size,kernel_size), in_channels => in_channels, pad = SamePad(), swish),
        BatchNorm(in_channels),
        Conv((1,1), in_channels => in_channels, swish, groups = 1, pad = SamePad(), stride = 1, bias = true),
        BatchNorm(in_channels),
        ),
        +
        )
    
    return layer
end



function main_model(N_classes::Int, no_blocks:: Int, patch_size::Int, in_channels::Int; kernel_size = 3::Int)
    patcher = Conv((patch_size, patch_size), 3=>in_channels, groups =  1, pad = SamePad(), stride = patch_size; bias  = false)
    main_mod = Chain(patcher,
    [block(in_channels,  kernel_size) for i in 1:no_blocks]...,
    AdaptiveMeanPool((1,1)),
            Flux.flatten,
            Dense(in_channels,N_classes)
    )
    return main_mod
end
    
@with_kw mutable struct Args
    η:: Float32 = 1e-3
    batch_size::Int64 = 64
    epochs:: Int32 = 100
    use_cuda:: Bool = true
    
end

function get_data(batch_size)
    X_train,y_train = MLDatasets.CIFAR100(split =:train).features, MLDatasets.CIFAR100(split =:train).targets.fine
    X_test,y_test = MLDatasets.CIFAR100(split =:test).features, MLDatasets.CIFAR100(split =:test).targets.fine
    train_data = DataLoader((X_train, y_train), batchsize= batch_size, shuffle = true, parallel = true)
    test_data = DataLoader((X_test, y_test), batchsize= batch_size, parallel = true)
    return train_data, test_data
end



function train()
    args = Args()   
    if args.use_cuda
        device = gpu
    else
        device = cpu        
    end
    ##### Local Variables ####
    acc_::Float32 = 0
    loss_::Float32 = 0
    #######- Def of the main model - ###########
    model = main_model(100, 25, 4, 128) |> device
    opt_state = Flux.setup(Momentum(), model)
        @info "Model initialized!"
    ####### - Get the training data - ##########
    train_data, test_data = get_data(args.batch_size)
    ###### - Loss Function - ############
    loss(x,y) = Flux.Losses.logitcrossentropy(x,y)
    ####### - Training loop - ###########
    @info "Training process is about to start!"
    @showprogress for epoch in 1:args.epochs
        for (x,y) in train_data
            y_oh = onehotbatch(y, 0:99)
            value, grad = Flux.withgradient(model) do m
                y_pred = m(x)
                loss(y_pred, y_oh)
            end
            Flux.update!(opt_state, model, grad[1])
        end

    end
    return model
end

if abspath(PROGRAM_FILE) == @__FILE__
    model = train()
    @save "mymodel.bson" model
end
