unet_hyperparam = {
    'batch_size': 8,
    'model_strides': [(2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2)],
    'model_features': [32, 64, 128, 256, 512, 1024]
}


def get_default_hyperparams(args):
    """Retrieve default hyperparameters for given neural network architecture"""

    # used for inference
    if isinstance(args, dict):
        if args['model_type'] == 'unet':
            args['model_strides'] = unet_hyperparam['model_strides']
            args['model_features'] = unet_hyperparam['model_features']
        args = type('_', (object,), args)

    # used at train-time
    else:
        if args.model_type == 'unet':
            args.batch_size = unet_hyperparam['batch_size']
            args.model_strides = unet_hyperparam['model_strides']
            args.model_features = unet_hyperparam['model_features']
    return args
