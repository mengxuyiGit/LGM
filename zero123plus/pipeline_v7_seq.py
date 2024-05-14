from typing import Any, Dict, Optional
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers

import numpy
import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.distributed
import transformers
from collections import OrderedDict
from PIL import Image
from torchvision import transforms
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer


import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    UNet2DConditionModel,
    ImagePipelineOutput
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.attention_processor import Attention, AttnProcessor, XFormersAttnProcessor, AttnProcessor2_0
from diffusers.utils.import_utils import is_xformers_available
from ipdb import set_trace as st
from custom_unet.unet_crossdomain_dev import UNet2DConditionModelCrossDomainAttn # as UNet2DConditionModel

def to_rgb_image(maybe_rgba: Image.Image):
    if maybe_rgba.mode == 'RGB':
        return maybe_rgba
    elif maybe_rgba.mode == 'RGBA':
        rgba = maybe_rgba
        img = numpy.random.randint(127, 128, size=[rgba.size[1], rgba.size[0], 3], dtype=numpy.uint8)
        img = Image.fromarray(img, 'RGB')
        img.paste(rgba, mask=rgba.getchannel('A'))
        return img
    else:
        raise ValueError("Unsupported image type.", maybe_rgba.mode)


class ReferenceOnlyAttnProc(torch.nn.Module):
    def __init__(
        self,
        chained_proc,
        enabled=False,
        name=None
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.chained_proc = chained_proc
        self.name = name

    def __call__(
        self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None,
        mode="w", ref_dict: dict = None, is_cfg_guidance = False
    ):
        # print(f"mode: {mode}, enabled: {self.enabled}, hidden_states: {getattr(hidden_states, 'shape', None)}, encoder_hidden_states: {getattr(encoder_hidden_states, 'shape', None)}")
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        if self.enabled and is_cfg_guidance:
            print("res0") # never come into this block
            res0 = self.chained_proc(attn, hidden_states[:1], encoder_hidden_states[:1], attention_mask)
            hidden_states = hidden_states[1:]
            encoder_hidden_states = encoder_hidden_states[1:]
        if self.enabled:
            if mode == 'w':
                ref_dict[self.name] = encoder_hidden_states
            elif mode == 'r':
                encoder_hidden_states = torch.cat([encoder_hidden_states, ref_dict.pop(self.name)], dim=1)
            elif mode == 'm':
                encoder_hidden_states = torch.cat([encoder_hidden_states, ref_dict[self.name]], dim=1)
            else:
                assert False, mode
        res = self.chained_proc(attn, hidden_states, encoder_hidden_states, attention_mask)
        # print("attention_mask", attention_mask), None
        if self.enabled and is_cfg_guidance:
            res = torch.cat([res0, res])
    
        return res

# ################################################################################################################################################################################
import einops
class XFormersJointAttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        num_tasks=5
    ):  
        
        input_ndim = hidden_states.ndim
       
        residual = hidden_states # torch.Size([5, 4096, 320])

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        
        # if input_ndim == 4:
        #     batch_size, channel, height, width = hidden_states.shape
        #     hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        # from yuancheng; here attention_mask is None
        if attention_mask is not None:
            # expand our mask's singleton query_tokens dimension:
            #   [batch*heads,            1, key_tokens] ->
            #   [batch*heads, query_tokens, key_tokens]
            # so that it can be added as a bias onto the attention scores that xformers computes:
            #   [batch*heads, query_tokens, key_tokens]
            # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        assert num_tasks == 5  # only support two tasks now

        key_0, key_1 = torch.chunk(key, dim=0, chunks=5)  # keys shape (b t) d c
        st()
        value_0, value_1 = torch.chunk(value, dim=0, chunks=2)
        key = torch.cat([key_0, key_1], dim=1)  # (b t) 2d c
        value = torch.cat([value_0, value_1], dim=1)  # (b t) 2d c
        key = torch.cat([key]*2, dim=0)   # ( 2 b t) 2d c
        value = torch.cat([value]*2, dim=0)  # (2 b t) 2d c
        st()
        
        key = einops.rearrange(key, "(B A) (V S) C -> (B V) (A S) C", B=1, A=5, V=8)
        value = einops.rearrange(value, "(B A) (V S) C -> (B V) (A S) C", B=1, A=5, V=8)

        
        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        
        return hidden_states

class CustomJointAttention(Attention):
    def set_use_memory_efficient_attention_xformers(
        self, use_memory_efficient_attention_xformers: bool, *args, **kwargs
    ):
        processor = XFormersJointAttnProcessor()
        self.set_processor(processor)
        # st()
        # print("using xformers attention processor")
        
from diffusers.models.attention import BasicTransformerBlock
# from diffusers.utils import BaseOutput, deprecate, maybe_allow_in_graph
from diffusers.models.attention import FeedForward, AdaLayerNorm, AdaLayerNormZero, Attention
if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None

from copy import deepcopy

# @maybe_allow_in_graph
class BasicTransformerBlockCrossDomainNoPosEmbed(nn.Module):
    def __init__(
        
        self,
        block: BasicTransformerBlock,
    ):
        super().__init__()
        self.only_cross_attention = block.only_cross_attention
        
        self.norm1 = block.norm1 
        self.attn1 = block.attn1
        
        self.norm_joint_mid = deepcopy(block.norm1)
        self.attn_joint_mid = deepcopy(block.attn1)
        # st() # check whether the block,.attn1.to_out[0] has been changed
        nn.init.zeros_(self.attn_joint_mid.to_out[0].weight.data)
    
        self.norm2 = block.norm2
        self.attn2 = block.attn2
        
        self.norm3 =  block.norm3
            
        self.ff = block.ff
       
        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0
    
    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int):
            # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim
    
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ):
        assert attention_mask is None # not supported yet
        
         # 0. Self-Attention
        norm_hidden_states = self.norm1(hidden_states)

        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            # num_views=self.num_views,
            # mvcd_attention=self.mvcd_attention,
            **cross_attention_kwargs,
        )
        
        
        hidden_states = attn_output + hidden_states
        
        if hidden_states.ndim == 4:
            st()
            hidden_states = hidden_states.squeeze(1)
        
        # joint attention twice
        # if self.cd_attention_mid:
        # st() # hidden_states.shape: torch.Size([5, 4096, 320])
        # hidden_states = einops.rearrange(hidden_states, "(B A) (V S) C -> (B V) (A S) C", B=1, A=5, V=8)
        hidden_states = einops.rearrange(hidden_states, "(B A) (V S) C -> (B V) (A S) C", A=5, V=8)
        # torch.Size([8, 2560, 320])
        norm_hidden_states = (
            self.norm_joint_mid(hidden_states) # timestamp if self.use_ada_layer_norm else self.norm_joint_mid(hidden_states)
        )
        hidden_states = self.attn_joint_mid(norm_hidden_states) + hidden_states
        # st() # torch.Size([8, 2560, 320])
        # hidden_states = einops.rearrange(hidden_states, "(B V) (A S) C -> (B A) (V S) C", B=1, A=5, V=8)
        hidden_states = einops.rearrange(hidden_states, "(B V) (A S) C -> (B A) (V S) C", A=5, V=8)
        # st() torch.Size([5, 4096, 320])
        
        # hidden_states.shape: torch.Size([5, 4096, 320])
        # 3. Cross-Attention
        norm_hidden_states = self.norm2(hidden_states)
        
        attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
        hidden_states = attn_output + hidden_states
        
        # 4. Feed-forward
        norm_hidden_states = self.norm3(hidden_states) # FIXME: original 2D  cond unet does not have a layerorm to be norm3
        
        ff_output = self.ff(norm_hidden_states)
        
        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            st()
            hidden_states = hidden_states.squeeze(1)

        return hidden_states

from diffusers.models.embeddings import SinusoidalPositionalEmbedding
# @maybe_allow_in_graph
class BasicTransformerBlockCrossDomainPosEmbed(nn.Module):
    def __init__(
        
        self,
        block: BasicTransformerBlock,
    ):
        super().__init__()
        self.only_cross_attention = block.only_cross_attention
        
        self.norm1 = block.norm1 
        self.attn1 = block.attn1
        
        self.norm_joint_mid = deepcopy(block.norm1)
        self.attn_joint_mid = deepcopy(block.attn1)
        # st() # check whether the block,.attn1.to_out[0] has been changed
        nn.init.zeros_(self.attn_joint_mid.to_out[0].weight.data)
    
        self.norm2 = block.norm2
        self.attn2 = block.attn2
        
        self.norm3 =  block.norm3
            
        self.ff = block.ff
       
        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

        # pos embed
        dim = self.norm1.normalized_shape[0] # 320 640 1280
        num_positional_embeddings = 4096
        self.pos_embed = SinusoidalPositionalEmbedding(dim, max_seq_length=num_positional_embeddings)
        
    
    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int):
            # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim
    
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ):
        assert attention_mask is None # not supported yet
        
         # 0. Self-Attention
        norm_hidden_states = self.norm1(hidden_states)

        
        if self.pos_embed is not None:
            # print("norm_hidden_states: ", norm_hidden_states.shape)
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            # num_views=self.num_views,
            # mvcd_attention=self.mvcd_attention,
            **cross_attention_kwargs,
        )
        
        
        hidden_states = attn_output + hidden_states
        
        if hidden_states.ndim == 4:
            st()
            hidden_states = hidden_states.squeeze(1)
        
        # joint attention twice
        # if self.cd_attention_mid:
        # st() # hidden_states.shape: torch.Size([5, 4096, 320])
        # hidden_states = einops.rearrange(hidden_states, "(B A) (V S) C -> (B V) (A S) C", B=1, A=5, V=8)
        hidden_states = einops.rearrange(hidden_states, "(B A) (V S) C -> (B V) (A S) C", A=5, V=8)
        # torch.Size([8, 2560, 320])
        norm_hidden_states = (
            self.norm_joint_mid(hidden_states) # timestamp if self.use_ada_layer_norm else self.norm_joint_mid(hidden_states)
        )

        if self.pos_embed is not None: # and self.norm_type != "ada_norm_single":
            # print("norm_hidden_states: ", norm_hidden_states.shape)
            norm_hidden_states = self.pos_embed(norm_hidden_states)
            
        hidden_states = self.attn_joint_mid(norm_hidden_states) + hidden_states
        # st() # torch.Size([8, 2560, 320])
        # hidden_states = einops.rearrange(hidden_states, "(B V) (A S) C -> (B A) (V S) C", B=1, A=5, V=8)
        hidden_states = einops.rearrange(hidden_states, "(B V) (A S) C -> (B A) (V S) C", A=5, V=8)
        # st() torch.Size([5, 4096, 320])
        
        # hidden_states.shape: torch.Size([5, 4096, 320])
        # 3. Cross-Attention
        norm_hidden_states = self.norm2(hidden_states)
        
        if self.pos_embed is not None: # and self.norm_type != "ada_norm_single":
            # print("norm_hidden_states: ", norm_hidden_states.shape)
            norm_hidden_states = self.pos_embed(norm_hidden_states)
        
        attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
        hidden_states = attn_output + hidden_states
        
        # 4. Feed-forward
        norm_hidden_states = self.norm3(hidden_states) # FIXME: original 2D  cond unet does not have a layerorm to be norm3
        
        ff_output = self.ff(norm_hidden_states)
        
        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            st()
            hidden_states = hidden_states.squeeze(1)

        return hidden_states

# def random_initialize_unet(unet):
#     # Recursively initialize weights of a given module

#     # Apply weight initialization
#     print("Random init UNet weights")
#     unet.apply(init_weights)

def init_weights(m):
    print("Random init UNet weights")
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def modify_unet(unet):
    # Recursive function to modify transformer blocks
    def replace_transformer_blocks(module):
        for name, child in module.named_children():
            if isinstance(child, BasicTransformerBlock):
                # print(name)
                # Replace the existing BasicTransformerBlock with a custom one
                setattr(module, name, BasicTransformerBlockCrossDomainPosEmbed(child))  # configure appropriately
            else:
                replace_transformer_blocks(child)

    # Apply modifications
    replace_transformer_blocks(unet)

class RefOnlyNoisedUNet(torch.nn.Module):
    def __init__(self, unet: UNet2DConditionModel, train_sched: DDPMScheduler, val_sched: EulerAncestralDiscreteScheduler):
        # -> None:
        super().__init__()
       
        with open("unet_6set_before_modify.txt", "w") as f:
            print(unet, file=f)
        
        self.duplicate_cond_lat = False
        self.train_sched = train_sched
        self.val_sched = val_sched
        self.is_generator = False
        
        #  set blocks
        modify_unet(unet=unet)

        unet_lora_attn_procs = dict()
        # for name, _ in unet.attn_processors.items():
        for name, _ in unet.attn_processors.items():
            # print(name)
            # if name.endswith("attn_joint_mid.processor"):
            #     default_attn_proc = XFormersJointAttnProcessor()
            # else:
            if torch.__version__ >= '2.0':
                default_attn_proc = AttnProcessor2_0()
            elif is_xformers_available():
                default_attn_proc = XFormersAttnProcessor()
            else:
                default_attn_proc = AttnProcessor()
            
            enabled = name.endswith("attn1.processor") # or name.endswith("attn_joint_mid.processor")
            unet_lora_attn_procs[name] = ReferenceOnlyAttnProc(
                default_attn_proc, enabled=enabled, name=name
            )
            # if name.endswith("attn_joint_mid.processor"):
            #     unet_lora_attn_procs[name] = XFormersJointAttnProcessor()
          
        # st()
        # unet.set_attn_processor(unet_lora_attn_procs)
        unet.set_attn_processor(unet_lora_attn_procs)
        
        # Random init unet weights
        unet.apply(init_weights)
       
        self.unet = unet
        with open("unet_7seq_after_modify.txt", "w") as f:
            print(self.unet, file=f)
        
    

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.unet, name)

    def forward_cond(self, noisy_cond_lat, timestep, encoder_hidden_states, class_labels, ref_dict, is_cfg_guidance, **kwargs):
        if is_cfg_guidance:
            encoder_hidden_states = encoder_hidden_states[1:]
            class_labels = class_labels[1:]
        # print("class_labels (forward_cond): ", class_labels)
        # from ipdb import set_trace as st; st()
        self.unet(
            noisy_cond_lat, timestep,
            encoder_hidden_states=encoder_hidden_states,
            class_labels=class_labels,
            cross_attention_kwargs=dict(mode="w", ref_dict=ref_dict),
            **kwargs
        )

    def forward(
        self, sample, timestep, encoder_hidden_states, class_labels=None,
        *args, cross_attention_kwargs,
        down_block_res_samples=None, mid_block_res_sample=None,
        **kwargs
    ):
        cond_lat = cross_attention_kwargs['cond_lat']
      
        if self.duplicate_cond_lat:
            cond_lat = cond_lat.repeat(1,5,1,1)
    
        is_cfg_guidance = cross_attention_kwargs.get('is_cfg_guidance', False)
        noise = torch.randn_like(cond_lat)
        if self.is_generator:
            # cond_timestep = torch.randint(500, 501, size=timestep.shape, device=timestep.device)
            cond_timestep = torch.zeros_like(timestep, dtype=timestep.dtype, device=timestep.device)
            # if self.training:
            #     cond_timestep = torch.randint(200, size=timestep.shape, device=timestep.device)
            # else:
            #     cond_timestep = torch.zeros_like(timestep, dtype=timestep.dtype, device=timestep.device)
        else:
            cond_timestep = timestep
        # cond_timestep = timestep
        if self.training:
            noisy_cond_lat = self.train_sched.add_noise(cond_lat, noise, cond_timestep)
            noisy_cond_lat = self.train_sched.scale_model_input(noisy_cond_lat, cond_timestep)
        else:
            noisy_cond_lat = self.val_sched.add_noise(cond_lat, noise, cond_timestep.reshape(-1))
            noisy_cond_lat = self.val_sched.scale_model_input(noisy_cond_lat, cond_timestep.reshape(-1))
        ref_dict = {}
        self.forward_cond(
            noisy_cond_lat, cond_timestep,
            encoder_hidden_states, class_labels,
            ref_dict, is_cfg_guidance, **kwargs
        )
        # print("class_labels (forward_cond in forward): ", class_labels)
        # from ipdb import set_trace as st; st()
        # print("ref_dict:\n",ref_dict)
        # st()
        weight_dtype = self.unet.dtype
       
        return self.unet(
            sample, timestep,
            encoder_hidden_states, *args,
            class_labels=class_labels,
            cross_attention_kwargs=dict(mode="r", ref_dict=ref_dict, is_cfg_guidance=is_cfg_guidance),
            down_block_additional_residuals=[
                sample.to(dtype=weight_dtype) for sample in down_block_res_samples
            ] if down_block_res_samples is not None else None,
            mid_block_additional_residual=(
                mid_block_res_sample.to(dtype=weight_dtype)
                if mid_block_res_sample is not None else None
            ),
            **kwargs
        )


def scale_latents(latents):
    latents = (latents - 0.22) * 0.75
    return latents


def unscale_latents(latents):
    latents = latents / 0.75 + 0.22
    return latents


def scale_image(image):
    image = image * 0.5 / 0.8
    return image


def unscale_image(image):
    image = image / 0.5 * 0.8
    return image


class DepthControlUNet(torch.nn.Module):
    def __init__(self, unet: RefOnlyNoisedUNet, controlnet: Optional[diffusers.ControlNetModel] = None, conditioning_scale=1.0) -> None:
        super().__init__()
        self.unet = unet
        if controlnet is None:
            self.controlnet = diffusers.ControlNetModel.from_unet(unet.unet)
        else:
            self.controlnet = controlnet
        DefaultAttnProc = AttnProcessor2_0
        if is_xformers_available():
            DefaultAttnProc = XFormersAttnProcessor
        self.controlnet.set_attn_processor(DefaultAttnProc())
        self.conditioning_scale = conditioning_scale

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.unet, name)

    def forward(self, sample, timestep, encoder_hidden_states, class_labels=None, *args, cross_attention_kwargs: dict, **kwargs):
        cross_attention_kwargs = dict(cross_attention_kwargs)
        control_depth = cross_attention_kwargs.pop('control_depth')
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=control_depth,
            conditioning_scale=self.conditioning_scale,
            return_dict=False,
        )
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_res_samples=down_block_res_samples,
            mid_block_res_sample=mid_block_res_sample,
            cross_attention_kwargs=cross_attention_kwargs
        )


class ModuleListDict(torch.nn.Module):
    def __init__(self, procs: dict) -> None:
        super().__init__()
        self.keys = sorted(procs.keys())
        self.values = torch.nn.ModuleList(procs[k] for k in self.keys)

    def __getitem__(self, key):
        return self.values[self.keys.index(key)]


class SuperNet(torch.nn.Module):
    def __init__(self, state_dict: Dict[str, torch.Tensor]):
        super().__init__()
        state_dict = OrderedDict((k, state_dict[k]) for k in sorted(state_dict.keys()))
        self.layers = torch.nn.ModuleList(state_dict.values())
        self.mapping = dict(enumerate(state_dict.keys()))
        self.rev_mapping = {v: k for k, v in enumerate(state_dict.keys())}

        # .processor for unet, .self_attn for text encoder
        self.split_keys = [".processor", ".self_attn"]

        # we add a hook to state_dict() and load_state_dict() so that the
        # naming fits with `unet.attn_processors`
        def map_to(module, state_dict, *args, **kwargs):
            new_state_dict = {}
            for key, value in state_dict.items():
                num = int(key.split(".")[1])  # 0 is always "layers"
                new_key = key.replace(f"layers.{num}", module.mapping[num])
                new_state_dict[new_key] = value

            return new_state_dict

        def remap_key(key, state_dict):
            for k in self.split_keys:
                if k in key:
                    return key.split(k)[0] + k
            return key.split('.')[0]

        def map_from(module, state_dict, *args, **kwargs):
            all_keys = list(state_dict.keys())
            for key in all_keys:
                replace_key = remap_key(key, state_dict)
                new_key = key.replace(replace_key, f"layers.{module.rev_mapping[replace_key]}")
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        self._register_state_dict_hook(map_to)
        self._register_load_state_dict_pre_hook(map_from, with_module=True)


class Zero123PlusPipeline(diffusers.StableDiffusionPipeline):
    tokenizer: transformers.CLIPTokenizer
    text_encoder: transformers.CLIPTextModel
    vision_encoder: transformers.CLIPVisionModelWithProjection

    feature_extractor_clip: transformers.CLIPImageProcessor
    unet: UNet2DConditionModel
    scheduler: diffusers.schedulers.KarrasDiffusionSchedulers

    vae: AutoencoderKL
    ramping: nn.Linear

    feature_extractor_vae: transformers.CLIPImageProcessor

    depth_transforms_multi = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        vision_encoder: transformers.CLIPVisionModelWithProjection,
        feature_extractor_clip: CLIPImageProcessor, 
        feature_extractor_vae: CLIPImageProcessor,
        ramping_coefficients: Optional[list] = None,
        safety_checker=None,
    ):
        DiffusionPipeline.__init__(self)

        self.register_modules(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
            unet=unet, scheduler=scheduler, safety_checker=None,
            vision_encoder=vision_encoder,
            feature_extractor_clip=feature_extractor_clip,
            feature_extractor_vae=feature_extractor_vae
        )
        self.register_to_config(ramping_coefficients=ramping_coefficients)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def prepare(self):
        train_sched = DDPMScheduler.from_config(self.scheduler.config)
        # self.scheduler = train_sched
        self.scheduler = DDIMScheduler.from_config(self.scheduler.config)
        if isinstance(self.unet, UNet2DConditionModel):
            self.unet = RefOnlyNoisedUNet(self.unet, train_sched, self.scheduler).eval()

    def add_controlnet(self, controlnet: Optional[diffusers.ControlNetModel] = None, conditioning_scale=1.0):
        self.prepare()
        self.unet = DepthControlUNet(self.unet, controlnet, conditioning_scale)
        return SuperNet(OrderedDict([('controlnet', self.unet.controlnet)]))

    def encode_condition_image(self, image: torch.Tensor):
        image = self.vae.encode(image).latent_dist.sample()
        return image

    @torch.no_grad()
    def prepare_conditions(self, image: Image.Image, depth_image: Image.Image = None, guidance_scale=4.0, prompt="", num_images_per_prompt=1):
        # image = to_rgb_image(image)
        image_1 = self.feature_extractor_vae(images=image, return_tensors="pt").pixel_values
        image_2 = self.feature_extractor_clip(images=image, return_tensors="pt").pixel_values
        if depth_image is not None and hasattr(self.unet, "controlnet"):
            depth_image = to_rgb_image(depth_image)
            depth_image = self.depth_transforms_multi(depth_image).to(
                device=self.unet.controlnet.device, dtype=self.unet.controlnet.dtype
            )
        image = image_1.to(device=self.vae.device, dtype=self.vae.dtype)
        image_2 = image_2.to(device=self.vae.device, dtype=self.vae.dtype)

        cond_lat = self.encode_condition_image(image)
        if guidance_scale > 1:
            negative_lat = self.encode_condition_image(torch.zeros_like(image))
            cond_lat = torch.cat([negative_lat, cond_lat])
        encoded = self.vision_encoder(image_2, output_hidden_states=False)
        global_embeds = encoded.image_embeds
        global_embeds = global_embeds.unsqueeze(-2)
        
        if hasattr(self, "encode_prompt"):
            encoder_hidden_states = self.encode_prompt(
                prompt,
                self.device,
                num_images_per_prompt,
                False
            )[0]
        else:
            encoder_hidden_states = self._encode_prompt(
                prompt,
                self.device,
                num_images_per_prompt,
                False
            )
        
    
        # from ipdb import set_trace as st; st()
        ramp = global_embeds.new_tensor(self.config.ramping_coefficients).unsqueeze(-1)
        encoder_hidden_states = encoder_hidden_states + global_embeds * ramp
        cak = dict(cond_lat=cond_lat)
        if hasattr(self.unet, "controlnet"):
            cak['control_depth'] = depth_image
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        
        # prompt_embeds = self._encode_prompt(
        #     None,
        #     # "sdfsf",
        #     device,
        #     num_images_per_prompt,
        #     do_classifier_free_guidance,
        #     negative_prompt=None,
        #     prompt_embeds=encoder_hidden_states,
        #     negative_prompt_embeds=None,
        #     lora_scale=None,
        # )

        # prompt_embeds_tuple = self.encode_prompt(
        #     prompt=prompt,
        #     device=device,
        #     num_images_per_prompt=num_images_per_prompt,
        #     do_classifier_free_guidance=do_classifier_free_guidance,
        #     negative_prompt=None,
        #     prompt_embeds=prompt_embeds,
        #     negative_prompt_embeds=None,
        #     lora_scale=None,
        # )

        prompt_embeds_tuple = self.encode_prompt(
            None,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds=encoder_hidden_states,
            negative_prompt_embeds=None,
            lora_scale=None,
        )
        
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])
        else:
            prompt_embeds = prompt_embeds_tuple[0]
         
        return prompt_embeds, cak

    @torch.no_grad()
    def __call__(
        self,
        image: Image.Image = None,
        prompt = "",
        *args,
        num_images_per_prompt: Optional[int] = 1,
        guidance_scale=4.0,
        depth_image: Image.Image = None,
        output_type: Optional[str] = "pil",
        width=640,
        height=960,
        num_inference_steps=28,
        return_dict=True,
        **kwargs
    ):
        self.prepare()
        if image is None:
            raise ValueError("Inputting embeddings not supported for this pipeline. Please pass an image.")
        assert not isinstance(image, torch.Tensor)
        prompt_embeds, cak = self.prepare_conditions(image, depth_image, guidance_scale, prompt)
        
        device = self._execution_device
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        generator = None
        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = torch.randn([1, num_channels_latents, height//self.vae_scale_factor, width//self.vae_scale_factor], device=device, dtype=prompt_embeds.dtype)
        # latents = torch.load("latents.pt").to(device, dtype=prompt_embeds.dtype)[:4]
        do_classifier_free_guidance = guidance_scale > 1.0
        # # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        # extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta=0.0)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cak,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # if do_classifier_free_guidance and guidance_rescale > 0.0:
                #     # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                #     noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    # if callback is not None and i % callback_steps == 0:
                    #     callback(i, t, latents)
        
        latents = unscale_latents(latents)
        if not output_type == "latent":
            image = unscale_image(self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0])
        else:
            image = latents

        image = self.image_processor.postprocess(image, output_type=output_type)
        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)