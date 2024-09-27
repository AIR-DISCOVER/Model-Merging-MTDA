_base_ = ['dtda.py']
uda = dict(
    alpha=0.998,
    imnet_feature_dist_lambda=0.005,
    imnet_feature_dist_classes=[6, 7, 11, 12, 13, 14, 15, 16, 17, 18],
    imnet_feature_dist_scale_min_ratio=0.75,
    debug_img_interval=500
)
