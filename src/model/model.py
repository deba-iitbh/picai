from monai.networks.nets import UNet, DynUNet, AttentionUnet

def neural_network_for_run(args, device):
    """Select neural network architecture for given run"""

    model = UNet(
        spatial_dims=len(args.image_shape),
        in_channels=args.num_channels,
        out_channels=args.num_classes,
        strides=args.model_strides,
        channels=args.model_features
    )

    model = model.to(device)
    print("Loaded Neural Network Arch.:", args.model_type)
    return model
