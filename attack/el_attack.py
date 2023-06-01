from visualizer import AAVisualizer


class ELAttack():
    def __init__(self,
                 cfg_file, 
                 ckpt_file, ) -> None:
        self.visualizer = AAVisualizer(cfg_file=cfg_file, ckpt_file=ckpt_file)
        self.model = self.visualizer.model

    def _forward(self, model, img):
        """Call different attack method to attack."""
        return self.test_attack(model, img)

    def test_attack(self, model, img):
        result = self.visualizer._forward(model=model, img=img)
        return result

