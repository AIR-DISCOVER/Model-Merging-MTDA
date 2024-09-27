# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# DAFormer w/o DSC in Tab. 7

_base_ = ['daformer_sepaspp_mitb5.py']

norm_cfg = dict(type='LN', requires_grad=True)
model = dict(
    decode_head=dict(
        decoder_params=dict(
            fusion_cfg=dict(
                norm_cfg=norm_cfg))))
