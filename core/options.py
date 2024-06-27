import tyro
from dataclasses import dataclass, field
from typing import Tuple, Literal, Dict, Optional, Union

# optimizer_config = {
#     'type': 'Adam',
#     'lr': 1e-3,
#     'weight_decay': 0.05,  # You can adjust this value as needed
# }

optimizer_config = {
    'type': 'Adam',
    'lr': 1e-2,
    'weight_decay': 0.,  # You can adjust this value as needed
}

splatter_optimizer_config = {
    'type': 'AdamW',
    'lr':  0.003,
    'weight_decay': 0.05,  # refer to lgm
    'betas' : (0.9, 0.95),
}


@dataclass
class Options:
    ### model
    # Unet image input size
    input_size: int = 256
    # Unet definition
    down_channels: Tuple[int, ...] = (64, 128, 256, 512, 1024, 1024)
    down_attention: Tuple[bool, ...] = (False, False, False, True, True, True)
    mid_attention: bool = True
    up_channels: Tuple[int, ...] = (1024, 1024, 512, 256)
    up_attention: Tuple[bool, ...] = (True, True, True, False)
    # Unet output size, dependent on the input_size and U-Net structure!
    splat_size: int = 64
    # gaussian render size
    output_size: int = 256

    ### dataset
    # data mode (only support s3 now)
    data_mode: Literal['s3', 'srn_cars'] = 's3'
    # fovy of the dataset
    fovy: float = 49.1
    # camera near plane
    znear: float = 0.5
    # camera far plane
    zfar: float = 2.5
    # number of all views (input + output)
    num_views: int = 12
    # number of views
    num_input_views: int = 4
    # camera radius
    cam_radius: float = 1.5 # to better use [-1, 1]^3 space
    # num workers
    num_workers: int = 8

    ### training
    # workspace
    workspace: str = './workspace'
    # resume
    resume: Optional[str] = None
    # batch size (per-GPU)
    batch_size: int = 8
    # gradient accumulation
    gradient_accumulation_steps: int = 1
    # training epochs
    num_epochs: int = 30
    # lpips loss weight
    lambda_lpips: float = 1.0
    # gradient clip
    gradient_clip: float = 1.0
    # mixed precision
    mixed_precision: str = 'bf16'
    # learning rate
    lr: float = 4e-4
    # augmentation prob for grid distortion
    prob_grid_distortion: float = 0.5
    # augmentation prob for camera jitter
    prob_cam_jitter: float = 0.5

    ### testing
    # test image path
    test_path: Optional[str] = None

    ### misc
    # nvdiffrast backend setting
    force_cuda_rast: bool = False
    # render fancy video with gaussian scaling effect
    fancy_video: bool = False

    
    ### optimize splatter
    fix_pretrained: bool = False
    eval_iter: int = 100
    save_iter: int = 200
    desc: Optional[str] = None
    # save_train_pred: bool = False
    data_path: Optional[str] = None
    
    ### zero123plus
    model_type: str = 'Zero123PlusGaussian'
    model_path: str = 'sudo-ai/zero123plus-v1.1' 
    custom_pipeline: str ='./zero123plus/pipeline_v2.py'
    bg: float = 0.5
    
    # use_rendering_loss: bool = False
    lambda_rendering: float = 1.0
    lambda_splatter: float = 1.0
    # use_splatter_loss: bool = False
    lambda_alpha: float = 1.0
    lambda_kl: float = 1e-6

    train_unet: bool = False
    discard_small_opacities: bool = False
    # for zero123++ debug
    render_gt_splatter: bool = False
    log_gs_loss_mse_dict: bool = False
    perturb_rot_scaling: bool = False
    attr_use_logrithm_loss: Tuple[str, ...] = ()

    # for LGM
    eval_fused_gt: bool = False
    eval_splatter_gt: bool = False
    use_adamW: bool = True
    lr_scheduler: str = 'Plat'
    lr_scheduler_patience: int = 2
    lr_scheduler_factor: float = 0.5
    min_lr_scheduled: float = 1e-6
    ## batch process
    scene_start_index: int = 0
    scene_end_index: int = -1
    early_stopping: bool = False
    early_stopping_patience: int = 10
    resume_workspace: Optional[str] = None
    verbose: bool = False
    
    # for zero123++ batch training 
    data_path_rendering: Optional[str] = None
    data_path_splatter_gt: Optional[str] = None
    data_path_vae_splatter: Optional[str] = None
    verbose_main: bool = False
    # scale_max_clamp: float = 
    plot_attribute_histgram: Tuple[str, ...] = ()
    gt_replace_pred: Tuple[str, ...] = ()
    set_random_seed: bool = False
    # biased_exp_scale_act: bool = False
    # biased_softplus_scale_act: bool = False
    scale_act: str = 'softplus'
    scale_act_bias: float = 0
    skip_predict_x0: bool = False
    scale_bias_learnable: bool = False
    scale_clamp_max: float = -2
    scale_clamp_min: float = -10
    normalize_scale_using_gt: Tuple[str, ...] = ()
    
    skip_training: bool = False
    save_train_pred: int = 50
    lr_schedule_by_train: bool = True
    check_splatter_alignment: bool = False
    
    decoder_mode: str = "v0_unfreeze_all" # "v1_fix_rgb" "v1_fix_rgb_remove_unscale" "v2_fix_rgb_more_conv"
    downsample_after_decode_latents: bool = False

    ## for code optimization
    init_from_mean: bool = False
    latent_resolution: int = 40
    
    optimizer: Dict[str, Union[str, float]] = field(default_factory=lambda: optimizer_config)
    splatter_optimizer: Dict[str, Union[str, float]] = field(default_factory=lambda: splatter_optimizer_config)
    code_init_from_0123_encoder: bool = True
    use_tanh_code_activation: bool = False
    splatter_guidance_interval: int = 1
    splatter_guidance_warmup: int = 500 
    
    overfit_one_scene: bool = False
    codes_from_encoder: bool = False
    decode_splatter_to_128: bool = False
    use_activation_at_downsample: bool = False
    downsample_mode: str = "DownsampleModule" # "AvgPool" "ConvAvgPool"
    decoder_upblocks_interpolate_mode:str = "last_layer" # "interpolate_upsample" "interpolate_downsample"
    replace_interpolate_with_avgpool: bool = False
    codes_from_diffusion: bool = False
    resume_unet: Optional[str] = None

    random_init_unet: bool = False
    codes_from_cache: bool = False # used to get cache latent stats distribution
    one_step_diffusion: Optional[int] = None
    code_cache_dir: Optional[str] = None
    lora_rank: int = 8
    render_input_views: bool = False
    lipschitz_coefficient: Optional[float] = None
    lipschitz_mode: Optional[str] = None # gaussian_noise, constant
    scheduler_type: Optional[str] = None
    save_ckpt_copies: bool = False
    mix_diffusion_interval: int = 10

    vae_on_splatter_image: bool = False

    ## change the splatter rep
    use_splatter_with_depth_offset: bool = False
    zero_init_xy_offset: bool = False
    always_zero_xy_offset: bool = False
    
    save_raw_tensor_splatter: bool = False
    reorganize_splatter_init: bool = False
    group_scale: bool = False


    # overfit eg3d data
    angle_y_step: float = 5.e-3
    further_optimize_splatter: bool = False
    workspace_to_save: Optional[str] = None
    splatter_to_encode: Optional[str] = None
    load_suffix: Optional[str] = None # "to_encode"
    load_ext: Optional[str] = None # "png" , "pt"
    optimization_objective: Optional[str] = None
    attr_group_mode: Optional[str] = None

    clip_image_to_encode: bool = False
    rendering_loss_on_splatter_to_encode: bool = False
    load_iter: int = 1000
    color_augmentation: bool = False
    loss_weights_decoded_splatter: float = 1.

    splatter_size: int = 128
    lambda_latent: float = 1
    attr_to_learn: Optional[str] = None
    fixed_noise_level: Optional[int] = None
    cd_spatial_concat: bool = False
    only_train_attention: bool = False
    rendering_loss_use_weight_t: bool = False

    finetune_decoder: bool = False
    max_train_steps: int = 30000
    resume_decoder: Optional[str] = None
    inference_finetuned_decoder: bool = False
    inference_finetuned_unet: bool = False
    guidance_scale: Optional[float] = 1.5
    log_each_attribute_loss: bool = False
    lambda_each_attribute_loss: Optional[Tuple[float,...]] = None
    train_unet_single_attr: Optional[Tuple[str, ...]] = None

    save_cond: bool = False
    render_video: bool = False
    lambda_splatter_lpips: float = 0.
    render_lgm_infer: Optional[Tuple[str, ...]] = None
    metric_GSO: bool = False

    decoder_with_domain_embedding: bool = False
    decoder_domain_embedding_mode: Optional[str] = None
    finetune_decoder_single_attr: Optional[Tuple[str, ...]] = None
    different_t_schedule: Optional[Tuple[int,...]] = None
    xyz_zero_t: bool = False

    calculate_FID: bool = False
    class_emb_cat: bool = False
    drop_cond_prob: float = 0.

    invalid_list: Optional[str] = None
    use_video_decoderST: bool = False

    cascade_on_xyz_opacity: bool = False
    

# all the default settings
config_defaults: Dict[str, Options] = {}
config_doc: Dict[str, str] = {}

config_doc['lrm'] = 'the default settings for LGM'
config_defaults['lrm'] = Options()

config_doc['small'] = 'small model with lower resolution Gaussians'
config_defaults['small'] = Options(
    input_size=256,
    splat_size=64,
    output_size=256,
    batch_size=8,
    gradient_accumulation_steps=1,
    mixed_precision='bf16',
)

config_doc['big'] = 'big model with higher resolution Gaussians'
config_defaults['big'] = Options(
    input_size=256,
    up_channels=(1024, 1024, 512, 256, 128), # one more decoder
    up_attention=(True, True, True, False, False),
    splat_size=128,
    output_size=512, # render & supervise Gaussians at a higher resolution.
    batch_size=8,
    num_views=8,
    gradient_accumulation_steps=1,
    mixed_precision='bf16',
)

config_doc['large'] = 'big model with even higher resolution Gaussians'
config_defaults['large'] = Options(
    input_size=256,
    up_channels=(1024, 1024, 512, 256, 128), # one more decoder
    up_attention=(True, True, True, False, False),
    splat_size=512,
    output_size=512, # render & supervise Gaussians at a higher resolution.
    batch_size=8,
    num_views=8,
    gradient_accumulation_steps=1,
    mixed_precision='bf16',
)

config_doc['tiny'] = 'tiny model for ablation'
config_defaults['tiny'] = Options(
    input_size=256, 
    down_channels=(32, 64, 128, 256, 512),
    down_attention=(False, False, False, False, True),
    up_channels=(512, 256, 128),
    up_attention=(True, False, False, False),
    splat_size=64,
    output_size=256,
    batch_size=16,
    num_views=8,
    gradient_accumulation_steps=1,
    mixed_precision='bf16',
)

AllConfigs = tyro.extras.subcommand_type_from_defaults(config_defaults, config_doc)
