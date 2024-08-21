import copy
import time
import torch
import numpy as np
import torch.nn as nn
from torchinfo import summary

from scripts.utils import *
from scripts.models import *
from scripts.utils import LoadJson


class CaloDiffu(nn.Module):
    """Diffusion based generative model"""
    def __init__(self, data_shape, config, NN_embed = None):
        super(CaloDiffu, self).__init__()
        self._data_shape = data_shape
        self.nvoxels = np.prod(self._data_shape)
        self.config = config if isinstance(config, dict) else LoadJson(config)
        self.num_heads=1
        self.nsteps = self.config["NSTEPS"]
        self.training_obj = self.config.get('TRAINING_OBJ', 'noise_pred')
        self.shower_embed = self.config.get('SHOWER_EMBED', '')
        self.fully_connected = ('FCN' in self.shower_embed)
        self.NN_embed = NN_embed

        supported = ['noise_pred', 'mean_pred', 'hybrid']
        is_obj = [s in self.training_obj for s in supported]
        
        if(not any(is_obj)):
            print("Training objective %s not supported!" % self.training_obj)
            exit(1)
        
        self.verbose = 1

        
        device = torch.device(self.config['DEVICE'])


        #Minimum and maximum maximum variance of noise
        self.beta_start = 0.0001
        self.beta_end = self.config.get("BETA_MAX", 0.02)

        #linear schedule
        schedd = self.config.get("NOISE_SCHED", "linear")
        self.discrete_time = True

        
        if("linear" in schedd):
            self.betas = torch.linspace(self.beta_start, self.beta_end, self.nsteps)
        elif("cosine" in schedd): 
            self.betas = cosine_beta_schedule(self.nsteps)
        elif("log" in schedd):
            self.discrete_time = False
            self.P_mean = -1.5
            self.P_std = 1.5
        else:
            print("Invalid NOISE_SCHEDD param %s" % schedd)
            exit(1)

        if(self.discrete_time):
            #precompute useful quantities for training
            self.alphas = 1. - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, axis = 0)

            #shift all elements over by inserting unit value in first place
            alphas_cumprod_prev = torch.nn.functional.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

            self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
            self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
            self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

            self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        self.time_embed = self.config.get("TIME_EMBED", 'sin')
        self.E_embed = self.config.get("COND_EMBED", 'sin')
        cond_dim = self.config['COND_SIZE_UNET']
        layer_sizes = self.config['LAYER_SIZE_UNET']
        block_attn = self.config.get("BLOCK_ATTN", False)
        mid_attn = self.config.get("MID_ATTN", False)
        compress_Z = self.config.get("COMPRESS_Z", False)


        if(self.fully_connected):
            #fully connected network architecture
            self.model = FCN(cond_dim = cond_dim, dim_in = self.config['SHAPE_ORIG'][1], num_layers = self.config['NUM_LAYERS_LINEAR'],
                    cond_embed = (self.E_embed == 'sin'), time_embed = (self.time_embed == 'sin') )

            self.R_Z_inputs = False

            summary_shape = [[1,config['SHAPE_ORIG'][1]], [1], [1]]


        else:
            RZ_shape = self.config['SHAPE_PAD'][1:]

            self.R_Z_inputs = self.config.get('R_Z_INPUT', False)
            self.phi_inputs = self.config.get('PHI_INPUT', False)

            in_channels = 1

            self.R_image, self.Z_image = create_R_Z_image(device, scaled = True, shape = RZ_shape)
            self.phi_image = create_phi_image(device, shape = RZ_shape)

            if(self.R_Z_inputs): in_channels = 3

            if(self.phi_inputs): in_channels += 1

            calo_summary_shape = list(copy.copy(RZ_shape))
            calo_summary_shape.insert(0, 1)
            calo_summary_shape[1] = in_channels

            calo_summary_shape[0] = 1
            summary_shape = [calo_summary_shape, [1, 1], [1, 1]]


            self.model = CondUnet(cond_dim = cond_dim, out_dim = 1, channels = in_channels, layer_sizes = layer_sizes, block_attn = block_attn, mid_attn = mid_attn, 
                    cylindrical =  self.config.get('CYLINDRICAL', False), compress_Z = compress_Z, data_shape = calo_summary_shape,
                    cond_embed = (self.E_embed == 'sin'), time_embed = (self.time_embed == 'sin')).to(device)

        print("\n\n Model: \n")
        summary(self.model, summary_shape)

    #wrapper for backwards compatability
    def load_state_dict(self, d):
        if('noise_predictor' in list(d.keys())[0]):
            d_new = dict()
            for key in d.keys():
                key_new = key.replace('noise_predictor', 'model')
                d_new[key_new] = d[key]
        else: d_new = d

        return super().load_state_dict(d_new)
    
    def RZ(self, x):
        batch_R_image = self.R_image.repeat([x.shape[0], 1,1,1,1]).to(device=x.device)
        batch_Z_image = self.Z_image.repeat([x.shape[0], 1,1,1,1]).to(device=x.device)

        return batch_R_image, batch_Z_image


    def add_RZPhi(self, x):
        cats = [x]
        if(self.R_Z_inputs):
            batch_R_image = self.R_image.repeat([x.shape[0], 1,1,1,1]).to(device=x.device)
            batch_Z_image = self.Z_image.repeat([x.shape[0], 1,1,1,1]).to(device=x.device)

            cats += [batch_R_image, batch_Z_image]

        if(self.phi_inputs):
            batch_phi_image = self.phi_image.repeat([x.shape[0], 1,1,1,1]).to(device=x.device)

            cats += [batch_phi_image]

        return torch.cat(cats, axis = 1) if len(cats) > 1 else x
    
    def noise_image(self, data = None, t = None, noise = None):

        if(noise is None): noise = torch.randn_like(data)

        if(t[0] <=0): return data

        if(self.discrete_time):
            sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, data.shape)
            sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, data.shape)
            out = sqrt_alphas_cumprod_t * data + sqrt_one_minus_alphas_cumprod_t * noise
            return out
        else:
            print("non discrete time not supported")
            exit(1)

    def compute_loss(self, data, energy, noise, t, loss_type = "l2", rnd_normal = None, energy_loss_scale = 1e-2):

        if(self.discrete_time): 
            x_noisy = self.noise_image(data, t, noise=noise)
            sigma = None
            sigma2 = extract(self.sqrt_one_minus_alphas_cumprod, t, data.shape)**2
        else:
            if rnd_normal is None:
                rnd_normal = torch.randn((data.size()[0],), device=data.device)
            sigma = (rnd_normal * self.P_std + self.P_mean).exp()
            x_noisy = data + torch.reshape(sigma, (data.shape[0], 1,1,1,1)) * noise
            sigma2 = sigma**2


        t_emb = torch.reshape(self.do_time_embed(t, self.time_embed, sigma), (-1, 1))
        pred = self.pred(x_noisy, energy, t_emb)

        weight = 1.
        x0_pred = None
        if('hybrid' in self.training_obj ):

            c_skip = torch.reshape(1. / (sigma2 + 1.), (data.shape[0], 1,1,1,1))
            c_out = torch.reshape(1./ (1. + 1./sigma2).sqrt(), (data.shape[0], 1,1,1,1))
            weight = torch.reshape(1. + (1./ sigma2), (data.shape[0], 1,1,1,1))

            #target = (data - c_skip * x_noisy)/c_out


            x0_pred = pred = c_skip * x_noisy + c_out * pred
            target = data

        elif('noise_pred' in self.training_obj):
            target = noise
            weight = 1.
            if('energy' in self.training_obj): 
                sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, data.shape)
                sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, data.shape)
                x0_pred = (x_noisy - sqrt_one_minus_alphas_cumprod_t * pred)/sqrt_alphas_cumprod_t
        elif('mean_pred' in self.training_obj):
            target = data
            weight = 1./ sigma2
            x0_pred = pred


        if loss_type == 'l1':
            loss = torch.nn.functional.l1_loss(target, pred)
        elif loss_type == 'l2':
            if('weight' in self.training_obj):
                loss = (weight * ((pred - data) ** 2)).sum() / (torch.mean(weight) * self.nvoxels)
            else:
                loss = torch.nn.functional.mse_loss(target, pred)

        elif loss_type == "huber":
            loss =torch.nn.functional.smooth_l1_loss(target, pred)
        else:
            raise NotImplementedError()

        if('energy' in self.training_obj):
            #sum total energy
            dims = [i for i in range(1,len(data.shape))]
            tot_energy_pred = torch.sum(x0_pred, dim = dims)
            tot_energy_data = torch.sum(data, dim = dims)
            loss_en = energy_loss_scale * torch.nn.functional.mse_loss(tot_energy_data, tot_energy_pred) / self.nvoxels
            loss += loss_en

        return loss
        


    def do_time_embed(self, t = None, embed_type = "identity",  sigma = None,):
        if(self.discrete_time):
            if(sigma is None): sigma = self.sqrt_one_minus_alphas_cumprod[t]

            if(embed_type == "identity" or embed_type == 'sin'):
                return t
            if(embed_type == "scaled"):
                return t/self.nsteps
            if(embed_type == "sigma"):
                return sigma.to(t.device)
            if(embed_type == "log"):
                return 0.5 * torch.log(sigma).to(t.device)
        else:
            if(embed_type == "log"):
                return 0.5 * torch.log(sigma).to(t.device)
            else:
                return sigma

    def pred(self, x, E, t_emb):

        if(self.NN_embed is not None): x = self.NN_embed.enc(x).to(x.device)
        out = self.model(self.add_RZPhi(x), E, t_emb)
        if(self.NN_embed is not None): out = self.NN_embed.dec(out).to(x.device)
        return out

    @torch.no_grad()
    def p_sample(self, x, E, t, noise=None, sample_algo='ddpm'):
        """Reverse the diffusion process (one step)"""

        if noise is None:
            noise = torch.randn(x.shape, device=x.device)

        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x.shape)
        posterior_variance_t = extract(self.posterior_variance, t, x.shape)

        t_emb = self.do_time_embed(t, self.time_embed)
        pred = self.pred(x, E, t_emb)

        if 'noise_pred' in self.training_obj:
            noise_pred = pred
            x0_pred = None

        elif'mean_pred' in self.training_obj:
            x0_pred = pred
            noise_pred = (x - sqrt_alphas_cumprod_t * x0_pred)/sqrt_one_minus_alphas_cumprod_t

        elif'hybrid' in self.training_obj:
            sigma2 = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)**2
            c_skip = 1. / (sigma2 + 1.)
            c_out = torch.sqrt(sigma2) / (sigma2 + 1.).sqrt()

            x0_pred = c_skip * x + c_out * pred
            noise_pred = (x - sqrt_alphas_cumprod_t * x0_pred)/sqrt_one_minus_alphas_cumprod_t

        if sample_algo == 'ddpm':
            # Sampling algo from https://arxiv.org/abs/2006.11239
            # Use results from our model (noise predictor) to predict the mean of posterior distribution of prev step
            post_mean = sqrt_recip_alphas_t * ( x - betas_t * noise_pred  / sqrt_one_minus_alphas_cumprod_t)
            out = post_mean + torch.sqrt(posterior_variance_t) * noise 
            if t[0] == 0: out = post_mean
        else:
            print("Algo %s not supported!" % sample_algo)
            exit(1)

        return out


    @torch.no_grad()
    def Sample(self, E, num_steps=200, sample_algo='ddpm', sample_offset=0, sample_step=1):
        """Generate samples from diffusion model.
        
        Args:
        E: Energies
        num_steps: The number of sampling steps. 
        Equivalent to the number of discretized time steps.    
        
        Returns: 
        Samples.
        """
        # Full sample (all steps)
        device = next(self.parameters()).device

        gen_size = E.shape[0]

        # start from pure noise (for each example in the batch)
        gen_shape = list(copy.copy(self._data_shape))
        gen_shape.insert(0, gen_size)

        # start from pure noise
        x = torch.randn(gen_shape, device=device)
        noise = x if 'fixed' in sample_algo else None

        time_steps = list(range(0, num_steps - sample_offset, sample_step))
        time_steps.reverse()
        
        t0 = time.time()
        for time_step in time_steps:      
            times = torch.full((gen_size,), time_step, device=device, dtype=torch.long)
            x = self.p_sample(x, E, times, noise=noise, sample_algo=sample_algo)

        print(f"Time for sampling {gen_size} events is {time.time() - t0} seconds")
        return x.detach().cpu().numpy()