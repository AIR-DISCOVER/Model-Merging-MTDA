uda = dict(
    type='DTDA',
    alpha=0.98,
    pseudo_threshold=0.968,
    pseudo_weight_ignore_top=0,
    pseudo_weight_ignore_bottom=0,
    imnet_feature_dist_lambda=0,
    imnet_feature_dist_classes=None,
    imnet_feature_dist_scale_min_ratio=None,
    mix='class',
    blur=True,
    color_jitter_strength=0.2,
    color_jitter_probability=0.2,
    debug_img_interval=1000,
    print_grad_magnitude=False,
    divergence_regulator=None,
    eval_model='teacher',
    share_encoder=False,
    merge_buffer=False,
    mask_mode=None,
    mask_alpha='same',
    mask_pseudo_threshold='same',
    mask_lambda=0,
    mask_generator=None,
    ema_method='naive',
    pretrained=False
)
use_ddp_wrapper = True