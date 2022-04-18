def load_model_state(net, args, checkpoint_path):
    state_dict = torch.load(
        checkpoint_path, map_location=torch.device(args.device))
    state_dict['args']['device'] = args.device
    params = net.state_dict()
    if 'model' in state_dict:
        model_state_dict = state_dict['model']
        cp_args = state_dict['args']
        cp_iter_num = state_dict['iter_num']
    else:
        model_state_dict = state_dict
        cp_args = None
        cp_iter_num = 0

    ignored_keys = ['maxiter', 'checkpoint_path', 'model_input_size', 't_total', 'num_workers',
                    'lr_warmup_ratio', 'lr_warmup_steps', 'local_rank', 'distributed', 'world_size',
                    'seed', 'debug', 'test_ds_name', 'test_ds_name', 'batch_size', 'dropout_prob',
                    'orig_patch_size', 'input_patch_size', 'D_pool_K', 'binarize',
                    'checkpoint_dir', 'iters', 'out_origsize', 'out_softscores', 'verbose_output',
                    'gpu', 'test_interp', 'do_remove_frag', 'reload_mask', 'saveiter',
                    'job_name', 'ds_names', 'train_ds_names', 'dice_warmup_steps', 'opt', 'lr', 'decay',
                    'grad_clip', 'localization_prob', 'tune_bn_only', 'MAX_DICE_W', 'deterministic',
                    'lr_schedule', 'out_fpn_do_dropout', 'randscale', 'do_affine', 'focus_class',
                    'bce_weight', 'D_scale', 'orig_input_size', 'input_scale',
                    'mean', 'std', 'mask_thres', 'use_pretrained', 'only_first_linear_in_squeeze',
                    'perturb_posw_range']

    # Some old models don't have these keys in args. But they use the values specified here.
    old_default_keys = {'out_fpn_upsampleD_scheme': 'interpolate',
                        'num_recurrences': 1, 'qk_have_bias': True}
    args2 = copy.copy(args)

    if args.net == 'segtran' and cp_args is not None:
        for k in old_default_keys:
            if k not in args:
                args2.__dict__[k] = old_default_keys[k]

        for k in cp_args:
            if (k not in ignored_keys) and (args2.__dict__[k] != cp_args[k]):
                print("args[{}]={}, checkpoint args[{}]={}, inconsistent!".format(
                    k, args2.__dict__[k], k, cp_args[k]))
                exit(0)

    params.update(model_state_dict)
    net.load_state_dict(params)
    del params
    del args2
    del state_dict
    del model_state_dict

