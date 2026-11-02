class LPIPS:  # stand-in: the encoder trainer imports training.losses, which imports lpips at module scope
    def __init__(self, *args, **kwargs):
        raise RuntimeError("lpips stand-in")
