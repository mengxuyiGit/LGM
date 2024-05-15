from core.options import Options
import os

def get_workspace_name(opt: Options, time_str: str, num_gpus: int):
    
    loss_str = 'loss'
    # assert (opt.lambda_rendering + opt.lambda_splatter + opt.lambda_lpips > 0), 'Must have at least one loss'
    if opt.lambda_rendering > 0:
        loss_str+=f'_render{opt.lambda_rendering}'
    elif opt.lambda_alpha > 0:
        loss_str+=f'_alpha{opt.lambda_alpha}'
    if opt.lambda_splatter > 0:
        loss_str+=f'_splatter{opt.lambda_splatter}'
    if opt.lambda_lpips > 0:
        loss_str+=f'_lpips{opt.lambda_lpips}'

    desc = opt.desc
    if opt.splatter_guidance_interval > 0:
        desc += f"-sp_guide_{opt.splatter_guidance_interval}"
    if opt.codes_from_encoder:
        desc += "-codes_from_encoder"
    else:
        optimizer_cfg = opt.optimizer.copy()
        desc += f"-codes_lr{optimizer_cfg['lr']}"
        
    desc += f"-{opt.decoder_mode}"
    if opt.decode_splatter_to_128:
        desc += "-pred128"
        if opt.decoder_upblocks_interpolate_mode is not None:
            desc += f"_{opt.decoder_upblocks_interpolate_mode}"
            if opt.decoder_upblocks_interpolate_mode!="last_layer" and opt.replace_interpolate_with_avgpool:
                desc += "_avgpool"
        
    ## the following may not exists, thus directly added to opt.desc if exists
    if len(opt.attr_use_logrithm_loss) > 0:
        loss_special = '-logrithm'
        for key in opt.attr_use_logrithm_loss:
            loss_special += f"_{key}"
        desc += loss_special
    
    if len(opt.normalize_scale_using_gt) > 0:
        loss_special = '-norm'
        for key in opt.normalize_scale_using_gt:
            loss_special += f"_{key}"
        desc += loss_special
        
    if opt.train_unet:
        desc += '-train_unet'
    if opt.skip_predict_x0:
        desc += '-skip_predict_x0'
    if opt.num_views != 20:
        desc += f'-numV{opt.num_views}'
    
    new_workspace = os.path.join(opt.workspace, f"{time_str}-{desc}-{loss_str}-lr{opt.lr}-{opt.lr_scheduler}")
    if opt.lr_scheduler == 'Plat':
            new_workspace += f"{opt.lr_scheduler_patience}"
    
    return new_workspace
    