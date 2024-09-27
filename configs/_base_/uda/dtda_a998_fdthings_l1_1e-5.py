_base_ = ['dtda_a998_fdthings.py']
uda = dict(
    divergence_regulator=dict(
        type='norm',  # _all_ = ['norm', 'cos']
        method='l1',  # _all_ = ['l1', 'l2'] if type == 'norm' else ['min', 'orth']
        weight=1e-5,
        _delete_=True),
)
