require 'torch'
require 'cudnn'
require 'cunn'
require 'image'
utils = require 'utils'

local t = require 'fb.resnet.torch/transforms'

-- For saving mat file
local matio = require 'matio'

-- Load the model
local model = torch.load('fb.resnet.torch/resnet-101.t7'):cuda()

-- Evaluate mode
model:evaluate()

-- The model was trained with this input normalization
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}

local transform = t.Compose{
   t.Scale(256),
   t.ColorNormalize(meanstd),
   t.CenterCrop(224),
}

local img = image.load('cat_dog.jpg', 3, 'float')
local size = img:size()
-- Scale, normalize, and crop the image
img = transform(img)

-- View as mini-batch of size 1
local batch = img:view(1, table.unpack(img:size():totable()))

-- Get the output
local output = model:forward(batch:cuda()):squeeze()

-- Take argmax
local score, pred_label = torch.max(output,1)

local weights = model.modules[11].weight
local activations = model.modules[8].output[1]
 
weights = weights[pred_label[1]]
weights = nn.utils.addSingletonDimension(weights)
weights = torch.reshape(weights,2048,1)

-- Summing and rectifying weighted activations across depth
local map = torch.sum(torch.cmul(activations, weights:view(activations:size(1), 1, 1):expandAs(activations)), 1)
gcam = map:cmul(torch.gt(map,0):typeAs(map))
gcam = image.scale(gcam:float(), size[3], size[2])
hm = utils.to_heatmap(gcam)
image.save('output/' .. tostring(pred_label[1]) .. '_resNet.png', image.toDisplayTensor(hm))
