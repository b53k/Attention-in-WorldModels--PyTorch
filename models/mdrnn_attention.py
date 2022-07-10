"""
Define MDRNN model, supposed to be used as a world model
on the latent space.
"""
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.normal import Normal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def gmm_loss(batch, mus, sigmas, logpi, reduce=True): # pylint: disable=too-many-arguments
    """ Computes the gmm loss.

    Compute minus the log probability of batch under the GMM model described
    by mus, sigmas, pi. Precisely, with bs1, bs2, ... the sizes of the batch
    dimensions (several batch dimension are useful when you have both a batch
    axis and a time step axis), gs the number of mixtures and fs the number of
    features.

    :args batch: (bs1, bs2, *, fs) torch tensor
    :args mus: (bs1, bs2, *, gs, fs) torch tensor
    :args sigmas: (bs1, bs2, *, gs, fs) torch tensor
    :args logpi: (bs1, bs2, *, gs) torch tensor
    :args reduce: if not reduce, the mean in the following formula is ommited

    :returns:
    loss(batch) = - mean_{i1=0..bs1, i2=0..bs2, ...} log(
        sum_{k=1..gs} pi[i1, i2, ..., k] * N(
            batch[i1, i2, ..., :] | mus[i1, i2, ..., k, :], sigmas[i1, i2, ..., k, :]))

    NOTE: The loss is not reduced along the feature dimension (i.e. it should scale ~linearily
    with fs).
    """
    batch = batch.unsqueeze(-2)
    normal_dist = Normal(mus, sigmas)
    g_log_probs = normal_dist.log_prob(batch)
    g_log_probs = logpi + torch.sum(g_log_probs, dim=-1)
    max_log_probs = torch.max(g_log_probs, dim=-1, keepdim=True)[0]
    g_log_probs = g_log_probs - max_log_probs

    g_probs = torch.exp(g_log_probs)
    probs = torch.sum(g_probs, dim=-1)

    log_prob = max_log_probs.squeeze() + torch.log(probs)
    if reduce:
        return - torch.mean(log_prob)
    return - log_prob

class _MDRNNBase(nn.Module):
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__()
        self.latents = latents
        self.actions = actions
        self.hiddens = hiddens
        self.gaussians = gaussians

        self.gmm_linear = nn.Linear(
            hiddens, (2 * latents + 1) * gaussians + 2)

    def forward(self, *inputs):
        pass

class MDRNN(_MDRNNBase):
    """ MDRNN model for multi steps forward """
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__(latents, actions, hiddens, gaussians)
        self.rnn_enc = nn.LSTM(latents + actions, hiddens)
        self.rnn_dec = nn.LSTM(hiddens, hiddens)
        self.fc = nn.Sequential(nn.Linear(self.hiddens*2, self.hiddens), nn.ReLU() ,nn.Linear(self.hiddens, 1))
        self.softmax = nn.Softmax(dim = 1)

    def attention(self, enc_hiddens, si):
        """ Computes context vector of size [batch, hidden_size] at a time"""
        raw_scores = torch.zeros((enc_hiddens.size(0),enc_hiddens.size(1),1), device = device) # <<<----------------------

        for i in range(len(enc_hiddens)):
            concat = torch.concat([si.squeeze(), enc_hiddens[i]], dim = -1)
            alignment_score = self.fc(concat) # scalar value [batch_size x 1]
            raw_scores[i] = alignment_score
        
        attention_weight = self.softmax(raw_scores)   # [32 x 16 x 1]
        context = torch.zeros((enc_hiddens.size(1), enc_hiddens.size(2)), device = device) # <<<----------------------

        for j in range(len(enc_hiddens)):
            context += attention_weight[j]*enc_hiddens[j]
        
        return context            # [16 x 256]          

    def forward(self, actions, latents, outputs): # pylint: disable=arguments-differ
        """ MULTI STEPS forward.

        :args actions: (SEQ_LEN, BSIZE, ASIZE) torch tensor
        :args latents: (SEQ_LEN, BSIZE, LSIZE) torch tensor

        :returns: mu_nlat, sig_nlat, pi_nlat, rs, ds, parameters of the GMM
        prediction for the next latent, gaussian prediction of the reward and
        logit prediction of terminality.
            - mu_nlat: (SEQ_LEN, BSIZE, N_GAUSS, LSIZE) torch tensor
            - sigma_nlat: (SEQ_LEN, BSIZE, N_GAUSS, LSIZE) torch tensor
            - logpi_nlat: (SEQ_LEN, BSIZE, N_GAUSS) torch tensor
            - rs: (SEQ_LEN, BSIZE) torch tensor
            - ds: (SEQ_LEN, BSIZE) torch tensor
        """
        seq_len, bs = actions.size(0), actions.size(1)

        ins = torch.cat([actions, latents], dim=-1)
        outs, (s0,_) = self.rnn_enc(ins)
        #============================== Attention ================================
        outputs = torch.zeros((seq_len, bs, outs.size(2)), device = device)  # <<<---------------------- 

        for i in range(len(outs)):
            context_i = self.attention(outs, s0)   #s0 = [1 x 16 x 256]
            output_i, (si,_) = self.rnn_dec(context_i.unsqueeze(0), (s0,_)) 
            outputs[i] = output_i.squeeze(0)
            s0 = si
        #=========================================================================
        gmm_outs = self.gmm_linear(outputs)

        stride = self.gaussians * self.latents

        mus = gmm_outs[:, :, :stride]
        mus = mus.view(seq_len, bs, self.gaussians, self.latents)

        sigmas = gmm_outs[:, :, stride:2 * stride]
        sigmas = sigmas.view(seq_len, bs, self.gaussians, self.latents)
        sigmas = torch.exp(sigmas)

        pi = gmm_outs[:, :, 2 * stride: 2 * stride + self.gaussians]
        pi = pi.view(seq_len, bs, self.gaussians)
        logpi = f.log_softmax(pi, dim=-1)

        rs = gmm_outs[:, :, -2]

        ds = gmm_outs[:, :, -1]

        return mus, sigmas, logpi, rs, ds

class MDRNNCellAttn(_MDRNNBase):
    """ MDRNN model for one step forward """
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__(latents, actions, hiddens, gaussians)
        #self.rnn = nn.LSTMCell(latents + actions, hiddens)
        self.rnn_enc = nn.LSTMCell(latents + actions, hiddens)
        self.rnn_dec = nn.LSTMCell(hiddens, hiddens)
        self.fc = nn.Sequential(nn.Linear(self.hiddens*2, self.hiddens), nn.Linear(self.hiddens, 1))
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, action, latent, hidden): # pylint: disable=arguments-differ
        """ ONE STEP forward.

        :args actions: (BSIZE, ASIZE) torch tensor
        :args latents: (BSIZE, LSIZE) torch tensor
        :args hidden: (BSIZE, RSIZE) torch tensor

        :returns: mu_nlat, sig_nlat, pi_nlat, r, d, next_hidden, parameters of
        the GMM prediction for the next latent, gaussian prediction of the
        reward, logit prediction of terminality and next hidden state.
            - mu_nlat: (BSIZE, N_GAUSS, LSIZE) torch tensor
            - sigma_nlat: (BSIZE, N_GAUSS, LSIZE) torch tensor
            - logpi_nlat: (BSIZE, N_GAUSS) torch tensor
            - rs: (BSIZE) torch tensor
            - ds: (BSIZE) torch tensor
        """
        in_al = torch.cat([action, latent], dim = 1)

        next_hidden = self.rnn_enc(in_al, hidden)
        out_enc = next_hidden[0]

        #---------------------------------------------------------------------------
        s0 = out_enc
        alignment = self.softmax(self.fc(torch.cat([s0, out_enc], dim = -1)))
        context = alignment*out_enc
        decoder_out = self.rnn_dec(context, (s0, hidden[1]))
        out_full = self.gmm_linear(decoder_out[0])
        #---------------------------------------------------------------------------
        #out_full = self.gmm_linear(out_enc)
        stride = self.gaussians * self.latents

        mus = out_full[:, :stride]
        mus = mus.view(-1, self.gaussians, self.latents)

        sigmas = out_full[:, stride:2 * stride]
        sigmas = sigmas.view(-1, self.gaussians, self.latents)
        sigmas = torch.exp(sigmas)

        pi = out_full[:, 2 * stride:2 * stride + self.gaussians]
        pi = pi.view(-1, self.gaussians)
        logpi = f.log_softmax(pi, dim=-1)

        r = out_full[:, -2]

        d = out_full[:, -1]

        return mus, sigmas, logpi, r, d, next_hidden
