require('paths')

local stringx = require('pl.stringx')
local file = require('pl.file')

function g_read_words(fname, vocab, ivocab)
    local data = file.read(fname)
    local lines = stringx.splitlines(data)
    local c = 0
    
    for n = 1, #lines do
        local w = stringx.split(lines[n])
        c = c + #w + 1
    end
    
    local words = torch.Tensor(c, 1)
    c = 0
  
    for n = 1, #lines do
        local w = stringx.split(lines[n])
        for i = 1, #w do
            c = c + 1
            if not vocab[w[i]] then
                ivocab[#vocab+1] = w[i]  
                vocab[w[i]] = #vocab + 1
            end
            words[c][1] = vocab[w[i]]
        end
        c = c + 1
        words[c][1] = vocab['<eos>']
    end
    
    print('Read ' .. c .. ' words from ' .. fname)
    return words
end  
     