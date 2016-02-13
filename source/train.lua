require 'torch'
require 'xlua'
require 'optim'

local t = require 'model'
local model = t.model
local fwmodel = t.model
local loss = t.loss

function nilling(module)
  module.gradBias = nil
  
  if module.finput then 
    model.finput = torch.Tensor()
  end

  module.gradWeight = nil
  module.output = torch.Tensor()
  
  if model.fgradInput then
    module.fgradInput = torch.Tensor()
  end
  
  module.gradInput = nil
end

function netLighter(network)
  nilling(network)
  if network.modules then
    for _,a in ipairs(network.modules) do
      netLighter(a)
    end
  end
end

local confusionMatrix = optim.ConfusionMatrix(classes)

local trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))

local w, dE_dW = model:getParameters()

local optimState = {
  learningRate = opt.learningRate,
  momentum = opt.momentum,
  weightDecay = opt.weightDecay,
  learningRateDecay = opt.learningRateDecay
}

local x = torch.Tensor(
opt.batchSize, 
trainData.data:size(2), 
trainData.data:size(3), 
trainData.data:size(4)
)

local yt = torch.Tensor(opt.batchSize)

if opt.type == 'cuda' then
  x = x:cuda()
  yt = yt:cuda()
end

local epoch

local function train(trainData)
  epoch = epoch or 1
  local time = sys.clock()
  local shuffle = torch.randperm(trainData:size())

  print('==> doing epoch on training data:') 
  print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

  for t=1,trainData:size(), opt.batchSize do
    xlua.progress(t, trainData:size())
    collectgarbage()

    if (t + opt.batchSize - 1) > trainData:size() then
      break
    end

    local idx = 1
    for i = t, t+opt.batchSize - 1 do
      x[idx] = trainData.data[shuffle[i]]
      yt[idx] = trainData.labels[shuffle[i]]
      idx = idx + 1
    end
    local eval_E = function(w)
      dE_dW:zero()

      local y = model:forward(x)
      local E = loss:forward(y, yt)

      local dE_dy = loss:backward(y, yt)
      model:backward(x, dE_dy)

      for i = 1, opt.batchSize do
        confusionMatrix:add(y[i], yt[i])
      end

      return E, dE_dw
    end

    optim.sgd(eval_E, w, optimState)
  end

  time = sys.clock() - time
  time = time / trainData:size()
  print("\n==> time to learn 1 sample = " .. (time * 1000) .. 'ms')

  print(confusionMatrix)

  trainLogger:add{['% mean class accuracy (train set)'] = confusionMatrix.totalValid * 100}
  if opt.plot then
    trainLogger:style{['% mean class accuracy (train set)'] = '-'}
    trainLogger:plot()
  end

  -- save/log current net
  local filename = paths.concat(opt.save, 'model.net')
  os.execute('mkdir -p ' .. sys.dirname(filename))
  print('==> saving model to '..filename)
  model1 = model:clone()
  netLighter(model1)
  torch.save(filename, model1)

  --next epoch
  confusionMatrix:zero()
  epoch = epoch + 1
end

return train

