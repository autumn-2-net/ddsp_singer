"""Shared model selection for training and inference."""


def build_singer(config, vocab_size):
    backbone = config.get('backbone', 'ddsp')
    if backbone == 'ddsp':
        from model.ddsp_singer import ddsp_singer
        return ddsp_singer(config=config, vocab_size=vocab_size)
    if backbone == 'nsf_cgmlp_vocos':
        from model.nsf_vocos_singer import NSFVocosSinger
        return NSFVocosSinger(config=config, vocab_size=vocab_size)
    raise ValueError(f'Unknown singer backbone: {backbone!r}')
