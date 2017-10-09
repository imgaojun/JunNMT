import torch
import torch.nn as nn

class GlobalAttention(nn.Module):
    def __init__(self, dim, attn_type="dot"):
        super(GlobalAttention, self).__init__()
        self.dim = dim
        self.attn_type = attn_type
        assert (self.attn_type in ["dot", "general", "mlp"]), (
                "Please select a valid attention type.")
        if self.attn_type == "general":
            self.linear_in = nn.Linear(dim, dim, bias=False)     

        # mlp wants it with bias
        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(dim*2, dim, bias=out_bias)         
                  
        self.sm = nn.Softmax()
        self.tanh = nn.Tanh()
        self.mask = None        

    def applyMask(self, mask):
        self.mask = mask

    def score(self, h_t, h_s):
        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t_ = h_t.view(tgt_batch*tgt_len, tgt_dim)
                h_t_ = self.linear_in(h_t_)
                h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            return torch.bmm(h_t, h_s_)

    def forward(self, input, context):
        batch, sourceL, dim = context.size()
        batch_, targetL, dim_ = input.size()

        # compute attention scores, as in Luong et al.
        align = self.score(input, context)

        # Softmax to normalize attention weights
        align_vectors = self.sm(align.view(batch*targetL, sourceL))
        align_vectors = align_vectors.view(batch, targetL, sourceL)

        # each context vector c_t is the weighted average
        # over all the source hidden states
        c = torch.bmm(align_vectors, context)

        # concatenate
        concat_c = torch.cat([c, input], 2).view(batch*targetL, dim*2)
        attn_h = self.linear_out(concat_c).view(batch, targetL, dim)
        if self.attn_type in ["general", "dot"]:
            attn_h = self.tanh(attn_h)

        attn_h = attn_h.transpose(0, 1).contiguous()
        align_vectors = align_vectors.transpose(0, 1).contiguous()

        # Check output sizes
        # targetL_, batch_, dim_ = attn_h.size()

        # targetL_, batch_, sourceL_ = align_vectors.size()
        return attn_h, align_vectors
