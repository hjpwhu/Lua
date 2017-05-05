local util, parent = torch.class('nn.LinearNB', 'nn.Module')

function util:__init(inputSize, outputSize)
   parent.__init(self)

   self.outputSize = outputSize
   self.inputSize = inputSize
   self.weight = torch.Tensor(outputSize, inputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)

   self:reset()
end

function util:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   if nn.oldSeed then
      for i=1,self.weight:size(1) do
         self.weight:select(1, i):apply(function()
            return torch.uniform(-stdv, stdv)
         end)
      end
   else
      self.weight:uniform(-stdv, stdv)
   end
end

function util:updateOutput(input)
   if input:dim() == 1 then
       self.output:resize(self.outputSize)
       self.output:zero()
       self.output:addmv(1, self.weight, input)
   elseif input:dim() == 2 then
       local nframe = input:size(1)
       local nunit = self.outputSize
       self.output:resize(nframe, nunit):zero()
       if not self.addBuffer or self.addBuffer:size(1) ~= nframe then
           self.addBuffer = input.new(nframe):fill(1)
       end
       if nunit == 1 then
           self.output:select(2,1):addmv(1, input, self.weight:select(1,1))
       else
           self.output:addmm(1, input, self.weight:t())
      end
   else
      error('input must be vector or matrix')
   end

   return self.output
end

function util:updateGradInput(input, gradOutput)
   if self.gradInput then
      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
      if input:dim() == 1 then
          self.gradInput:addmv(0, 1, self.weight:t(), gradOutput)
      elseif input:dim() == 2 then
          self.gradInput:addmm(0, 1, gradOutput, self.weight)
      end

      return self.gradInput
   end
end

function util:accGradParameters(input, gradOutput, scale)
   scale = scale or 1

   if input:dim() == 1 then
      self.gradWeight:addr(scale, gradOutput, input)
   elseif input:dim() == 2 then
      local nunit = self.outputSize
      if nunit == 1 then
         self.gradWeight:select(1,1):addmv(scale, input:t(), gradOutput:select(2,1))
      else
         self.gradWeight:addmm(scale, gradOutput:t(), input)
      end
   end
end

util.sharedAccUpdateGradParameters = util.accUpdateGradParameters