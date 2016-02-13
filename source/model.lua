require 'torch'
require 'image'
require 'nn'
require 'cunn'
require 'cutorch'

nn.SpatialConvolutionMM = nn.SpatialConvolution

local nOutputs   = 2
local nFeatures  = 1
local width      = 32
local height     = 32

local nStates    = {16, 32}
local filterSize = {5, 7}
local poolSize   = 4


local CNN = nn.Sequential()
CNN:add(nn.SpatialConvolutionMM(nFeatures, nStates[1], filterSize[1], filterSize[1]))
CNN:add(nn.Threshold())
CNN:add(nn.SpatialMaxPooling(poolSize, poolSize, poolSize, poolSize))

CNN:add(nn.SpatialConvolutionMM(nStates[1], nStates[2], filterSize[2], filterSize[2]))
CNN:add(nn.Threshold())

local classifier = nn.Sequential()

classifier:add(nn.Reshape(nStates[2]))
classifier:add(nn.Linear(nStates[2], 2))

classifier:add(nn.LogSoftMax())

for _,layer in ipairs(CNN.modules) do
  if layer.bias then
    layer.bias:fill(.2)
    if i == #CNN.modules - 1 then
      layer.bias:zero()
    end
  end
end

local model = nn.Sequential()
model:add(CNN)
model:add(classifier)

loss = nn.ClassNLLCriterion()

model = model:cuda()
loss = loss:cuda()

print(model)

return {
  model = model,
  loss = loss,
}
