local utils = {}

-- Preprocess the image before passing it to a Caffe model.
function utils.preprocess(path, width, height)
  local width = width or 224
  local height = height or 224

  -- load image
  local orig_image = image.load(path)

  -- handle greyscale and rgba images
  if orig_image:size(1) == 1 then
    orig_image = orig_image:repeatTensor(3, 1, 1)
  elseif orig_image:size(1) == 4 then
    orig_image = orig_image[{{1,3},{},{}}]
  end

  -- get the dimensions of the original image
  local im_height = orig_image:size(2)
  local im_width = orig_image:size(3)

  -- scale and subtract mean
  local img = image.scale(orig_image, width, height):double()
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  img = img:index(1, torch.LongTensor{3, 2, 1}):mul(255.0)
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img:add(-1, mean_pixel)
  return img, im_height, im_width
end

function utils.to_heatmap(map)
  map = image.toDisplayTensor(map)
  local cmap = torch.Tensor(3, map:size(2), map:size(3)):fill(1)
  for i = 1, map:size(2) do
    for j = 1, map:size(3) do
      local value = map[1][i][j]
      if value <= 0.25 then
        cmap[1][i][j] = 0
        cmap[2][i][j] = 4*value
      elseif value <= 0.5 then
        cmap[1][i][j] = 0
        cmap[3][i][j] = 2 - 4*value
      elseif value <= 0.75 then
        cmap[1][i][j] = 4*value - 2
        cmap[3][i][j] = 0
      else
        cmap[2][i][j] = 4 - 4*value
        cmap[3][i][j] = 0
      end
    end
  end
  return cmap
end

return utils
