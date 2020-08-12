from torch.utils.tensorboard import SummaryWriter

class _Tensorboard(object):
    def __init__(self, path):
        self.writer = SummaryWriter(path)
        pass


# for n_iter in range(100):
#     writer.add_scalar('Loss/train', np.random.random(), n_iter)
#     writer.add_scalar('Loss/test', np.random.random(), n_iter)
#     writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
#     writer.add_scalar('Accuracy/test', np.random.random(), n_iter)