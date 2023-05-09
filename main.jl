using CSV
using DataFrames
using Flux

using BSON: @save
using ProgressMeter: @showprogress

function NeuralNetwork()
    return Chain(
        Dense(478, 1,relu),
        #Dense(500,1,x->σ.(x))
    )
end

dataframe = CSV.read("test.csv", DataFrame)
frametoarray(dfs) = cat(Matrix.(dfs)..., dims=3)

placements = dataframe[:, 2:478]
grades = dataframe[:, 1:1]
data = Flux.Data.DataLoader((hcat(eachrow(placements)...), grades, batchsize=100,shuffle=true);
m    = NeuralNetwork()
opt = Descent(0.05)

loss(x, y) = sum(Flux.Losses.binarycrossentropy(m(x), y))
ps = Flux.params(m)
epochs = 20
println("starting training")
@showprogress for i in 1:epochs
    Flux.train!(loss, ps, data, opt)
end
@save "model.bson" model