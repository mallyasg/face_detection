require 'torch'
require 'image'
require 'nnx'
require 'cutorch'

-- Set the default tensor type to Float tensor
torch.setdefaulttensortype('torch.FloatTensor')
-- Load the images from disk to torch tensors
local imagesAll = torch.Tensor(41267, 3, 32, 32)
local labelsAll = torch.Tensor(41267)

-- Define the two classes
-- Numerically face corresponds to value 1 and background to 2
classes = {'face', 'background'}

print('==> Loading the images from file')
-- Load the background images
for f=0, 28033 do
  imagesAll[f + 1] = image.load('../data/face-dataset/bg/bg_' ..f.. '.png')
  labelsAll[f + 1] = 2
end

-- Load the face images
for f=28043, 41266 do
  imagesAll[f + 1] = image.load('../data/face-dataset/face/face_' ..f.. '.png')
  labelsAll[f + 1] = 1
end

-- Randomly shuffle the dataset 
local indices = torch.randperm((#labelsAll)[1])

local trainPercentage = 0.8

local trainSize = torch.floor(indices:size(1) * trainPercentage)
local testSize = indices:size(1) - trainSize

-- Create the train data
trainData = {
  data = torch.Tensor(trainSize, 1, 32, 32),
  labels = torch.Tensor(trainSize),
  size = function() return trainSize end
}

-- Create the test data
testData = {
  data = torch.Tensor(testSize, 1, 32, 32),
  labels = torch.Tensor(testSize),
  size = function() return testSize end
}

print('==> Load the images onto data tensors')
for i=1, trainSize do
  trainData.data[i] = imagesAll[indices[i]][1]:clone()
  trainData.labels[i] = labelsAll[indices[i]]
end

for i=trainSize + 1, testSize + trainSize do
  testData.data[i - trainSize] = imagesAll[indices[i]][1]:clone()
  testData.labels[i - trainSize] = labelsAll[indices[i]]
end

-- trainData.data = trainData.data:cuda()
-- trainData.labels = trainData.labels:cuda()

-- testData.data = testData.data:cuda()
-- testData.data = testData.data:cuda()

local channels = {'y'} --, 'u', 'v'}

local mean = {}
local stdv = {}

print('==> Compute the mean and standard deviation of the training data')
-- Compute the mean and standard deviation for each channel
-- and normalize the training data
for i, channel in ipairs(channels) do
  mean[i] = trainData.data[{{}, i, {}, {}}]:mean()
  stdv[i] = trainData.data[{{}, i, {}, {}}]:std()

  trainData.data[{{}, i, {}, {}}]:add(-mean[i])
  trainData.data[{{}, i, {}, {}}]:div(stdv[i])
end

-- Normalize the test/evaluation data using the mean and standard
-- deviation captured previously
for i, channel in ipairs(channels) do
  testData.data[{{}, i, {}, {}}]:add(-mean[i])
  testData.data[{{}, i, {}, {}}]:div(stdv[i])
end

print('==> Perform contrastive normalization')
-- Perform local contrast normalization
-- Define the neighborhood for contrast normalization
local neighborhood = image.gaussian1D(5)

local normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()
-- normalization = normalization:cuda()

-- Normalize all the channels
for i, channel in ipairs(channels) do
  for j=1, trainData:size() do
    trainData.data[{j, {i}, {}, {}}] = normalization:forward(trainData.data[{j, {i}, {}, {}}])
  end
  for j=1, testData:size() do
    testData.data[{j, {i}, {}, {}}] = normalization:forward(testData.data[{j, {i}, {}, {}}])
  end
end

-- Check if the mean and standard deviation of the train data and test data are 0 and 1
for i, channel in ipairs(channels) do
  local trainMean = trainData.data[{{}, i, {}, {}}]:mean()
  local trainStdv = trainData.data[{{}, i, {}, {}}]:std()
  
  local testMean = testData.data[{{}, i, {}, {}}]:mean()
  local testStdv = testData.data[{{}, i, {}, {}}]:std()

  print('Training data, '..channel..'-channel, mean: '..trainMean)
  print('Training data, '..channel..'-channel, stdv: '..trainStdv)

  print('Testing data, '..channel..'-channel, mean: '..testMean)
  print('Testing data, '..channel..'-channel, stdv: '..testStdv)
end

-- local first256Samples_y = trainData.data[{ {1,256},1 }]
-- image.display{image=first256Samples_y, nrow=16, legend='Some training examples: Y channel'}
-- local first256Samples_y = testData.data[{ {1,256},1 }]
-- image.display{image=first256Samples_y, nrow=16, legend='Some testing examples: Y channel'}

return {
  trainData = trainData,
  testData = testData,
  mean = mean,
  stdv = stdv,
  classes = classes
}
