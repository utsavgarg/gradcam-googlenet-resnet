require 'torch'
require 'nn'
require 'lfs'
require 'image'
utils = require 'utils'

torch.manualSeed(123)
torch.setdefaulttensortype('torch.FloatTensor')

require 'cunn'
require 'cutorch'
cutorch.setDevice(1)
cutorch.manualSeed(123)

-- For saving mat file
local matio = require 'matio'

-- Load CNN
local googlenet = dofile('inception.torch/googlenet.lua')
local cnn = googlenet({
        cudnn.SpatialConvolution,
        cudnn.SpatialMaxPooling,
        cudnn.ReLU,
        cudnn.SpatialCrossMapLRN
})

-- Set to evaluate and remove softmax layer
cnn:evaluate()
cnn:remove()
cnn:cuda()

-- Load image
local img = image.load('cat_dog.jpg', 3, 'float')
local size = img:size()
local img = utils.preprocess('cat_dog.jpg', 224, 224)

img = nn.utils.addSingletonDimension(img)
img = img:cuda()

-- predict
local output = cnn:forward(img)
output = output:squeeze()

-- Take argmax
local score, pred_label = torch.max(output,1)
local weights = cnn.modules[24].weight
local activations = cnn.modules[21].output[1]
weights = weights[pred_label[1]]
weights = nn.utils.addSingletonDimension(weights)
weights = torch.reshape(weights,1024,1)

-- Summing and rectifying weighted activations across depth
local map = torch.sum(torch.cmul(activations, weights:view(activations:size(1), 1, 1):expandAs(activations)), 1)
gcam = map:cmul(torch.gt(map,0):typeAs(map))
gcam = image.scale(gcam:float(), size[3], size[2])
hm = utils.to_heatmap(gcam)
image.save('output/' .. tostring(pred_label[1]) .. '_googleNet.png', image.toDisplayTensor(hm))
