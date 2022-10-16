import numpy as np

def z_score_norm(image, percentile):
    """
    Z-score normalization (mean=0; stdev=1), where intensities
    below or above the given percentile are discarded.
    """
    image = image.astype(np.float32)

    if percentile is not None:
        # clip distribution of intensity values
        lower_bnd = np.percentile(image, 100-percentile)
        upper_bnd = np.percentile(image, percentile)
        image = np.clip(image, lower_bnd, upper_bnd)

    # perform z-score normalization
    mean = np.mean(image)
    std = np.std(image)
    if std > 0:
        return (image - mean) / std
    else:
        return image * 0.
