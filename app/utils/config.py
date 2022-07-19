# default brain settings
APPLICATION_TITLE = "Unswell"
BRAIN_SMOOTHNESS = 500
BRAIN_OPACITY = 0.2
BRAIN_COLORS = [(1.0, 0.9, 0.9)]  # RGB percentages
BRAIN_CMAP = '' # bone, nipy_spectral, gist_earth, gist_stern may work best. left empty for black and white MRI
# default mask settings
MASK_SMOOTHNESS = 500

# MASK_COLORS = {'load': [(0, 0.12, 0.25), (0.22, 0.8, 0.8), (0, 0, 0), (0, 0.45, 0.85)],
#                'segment': [(0.52, 0.08, 0.29), (0.69, 0.05, 0.79), (0, 0, 0), (0.94, 0.07, 0.75)],
#                'predict': [(1.00, 0.25, 0.21), (1.00, 0.86, 0.00), (0, 0, 0), (1.00, 0.52, 0.11)]}

MASK_COLORS = {'Ground truth': [(1.00, 0.52, 0.11), (1.00, 0.25, 0.21), (0, 0, 0), (1.00, 0.86, 0.00)],
               'Segmented': [(0,0.78, 0.28), (0.1, 0.45, 0.33), (0,0,0), (0.34, 0.91, 0.78)]}


MASK_OPACITY = 1.0
MODEL_PATH = 'app/model_controller/weights/generator_epoch100_new.pth'

# model config

segtran_config = {
    'D_groupsize': 1,
    'D_pool_K': 2,
    'ablate_multihead': False,
    'attention_mode_dim': -1,
    'attn_clip': 500,
    'aug_degree': 0.5,
    'backbone_type': 'i3d',
    'base_initializer_range': 0.02,
    'batch_size': 1,
    'bb_feat_upsize': True,
    'binarize': False,
    'chosen_modality': -1,
    'debug': False,
    'device': 'cuda',
    'ds_class': 'BratsSet',
    'ds_split': 'all',
    'eval_robustness': False,
    'gpu': '0',
    'has_FFN_in_squeeze': False,
    'in_fpn_layers': '34',
    'in_fpn_scheme': 'AN',
    'in_fpn_use_bn': False,
    'inchan_to3_scheme': 'bridgeconv',
    'input_patch_size': [112, 112, 96],
    'input_scale': (1, 1, 1),
    'iters': '',
    'job_name': 'brats-2021valid',
    'mid_type': 'shared',
    'net': 'segtran',
    'num_attractors': 512,
    'num_classes': 4,
    'num_modes': 4,
    'num_translayers': 1,
    'orig_in_channels': 4,
    'orig_input_size': None,
    'orig_patch_size': (112, 112, 96),
    'out_fpn_layers': '1234',
    'out_fpn_scheme': 'AN',
    'out_fpn_upsampleD_scheme': 'conv',
    'pos_bias_radius': 7,
    'pos_code_every_layer': True,
    'pos_code_type': 'lsinu',
    'pos_code_weight': 1.0,
    'pos_in_attn_only': False,
    'qk_have_bias': True,
    'segtran_type': '3d',
    'task_name': 'brats',
    'test_interp': None,
    'trans_output_type': 'private',
    'translayer_compress_ratios': [1, 1],
    'use_pretrained': True,
    'use_squeezed_transformer': True,
    'verbose_output': False,
    'vis_mode': None,
    'xyz_permute': None,
}

unet_config={
    "num_classes": 4,
    "bce_weight":  [0, 3, 1, 1.75],
    "chosen_modality": -1,
    "orig_patch_size": [112, 112, 96],
    "input_patch_size": [112, 112, 96],
    "input_scale":     [1,   1,   1],
    "D_pool_K":         2,
    "localization_prob": 0.5,
}
