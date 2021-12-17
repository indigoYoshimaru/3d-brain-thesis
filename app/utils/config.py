# default brain settings
APPLICATION_TITLE = "Unswell"
BRAIN_SMOOTHNESS = 500
BRAIN_OPACITY = 0.2
BRAIN_COLORS = [(1.0, 0.9, 0.9)]  # RGB percentages

# default mask settings
MASK_SMOOTHNESS = 500
# MASK_COLORS = [(1, 0, 0),
#                (0, 1, 0),
#                (1, 1, 0),
#                (0, 0, 1),
#                (1, 0, 1),
#                (0, 1, 1),
#                (1, 0.5, 0.5),
#                (0.5, 1, 0.5),
#                (0.5, 0.5, 1)]  # RGB percentages

# MASK_COLORS = {'loaded': [(1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1)],
#                'segmented': [(1, 0, 1),(0, 1, 1),(1, 0.5, 0.5),],
#                'predicted': []}

MASK_COLORS = [(1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1)]

MASK_OPACITY = 1.0
MODEL_PATH = 'app/model_controller/weights/segtran_iter_81500.pth'
# model config

model_config = {
    'num_classes': 4,
    'backbone_type': 'i3d',
    'use_pretrained': True,
    'bb_feat_upsize': None,
    'bb_feat_dims': None,
    'in_fpn_use_bn': None,
    'use_squeezed_transformer': True,
    'num_attractors': 512,
    'num_translayers': 1,
    'num_modes': {'234': 2,  '34': 4,  '4': 4},
    'trans_output_type': 'private',
    'mid_type': 'shared',
    'pos_code_every_layer': None,
    'pos_in_attn_only': None,
    'base_initializer_range': 0.02,
    'pos_code_type': 'lsinu',
    'pos_code_weight': 1.0,
    'pos_bias_radius': 7,
    'ablate_multihead': False,
    'has_FFN_in_squeeze': False,
    'attn_clip': 500,
    'orig_in_channels': 4,
    'inchan_to3_scheme': 'bridgeconv',
    'D_groupsize': 1,
    'D_pool_K': 2,
    'out_fpn_upsampleD_scheme': 'conv',
    'input_scale':  (1,   1,   1),
    'device': 'cpu',
    'in_fpn_layers': '34',
    'out_fpn_layers': '1234',
    'in_fpn_scheme': 'AN',
    'out_fpn_scheme': 'AN',
    'translayer_compress_ratios': [1, 1],
    'attention_mode_dim': -1,
    'in_feat_dim': 1024,
    'feat_dim': 1024

}

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
    'device': 'cpu',
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
