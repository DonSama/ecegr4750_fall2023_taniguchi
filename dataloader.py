import torch
import math

# modified our previous CustomDataLoader to accept data extracted from images and stream data to our model from the disk.
# code largely borrowed from https://visualstudiomagazine.com/Articles/2021/04/01/pytorch-streaming.aspx?Page=1
class CustomDataloader():
    def __init__(self, x: torch.Tensor, y: torch.Tensor, batch_size: int = 1, buffer_size: int = 10, randomize: bool = False):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.randomize = randomize
        self.iter = None
        self.num_batches_per_epoch = math.ceil(self.get_length() / self.batch_size)
        self.buffer = []

    def get_length(self):
        return self.x.shape[0]

    def randomize_dataset(self):
        indices = torch.randperm(self.x.shape[0])
        self.x = self.x[indices]
        self.y = self.y[indices]

    def generate_iter(self):
        if self.randomize:
            self.randomize_dataset()

        batches = []
        for b_idx in range(self.num_batches_per_epoch):
            batches.append(
                {
                    'x_batch': self.x[b_idx * self.batch_size : (b_idx+1) * self.batch_size],
                    'y_batch': self.y[b_idx * self.batch_size : (b_idx+1) * self.batch_size],
                    'batch_idx': b_idx,
                }
            )
        self.iter = iter(batches)

    def fill_buffer(self):
        for _ in range(self.buffer_size):
            try:
                self.buffer.append(next(self.iter))
            except StopIteration:
                # Regenerate iterator for the next epoch
                self.generate_iter()
                self.buffer.append(next(self.iter))

    def fetch_batch(self):
        if not self.buffer:
            self.fill_buffer()

        return self.buffer.pop(0)