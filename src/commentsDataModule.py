

from commentsDataset import CommentsDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader

class CommentsDataModule(pl.LightningDataModule):
    def __init__(self, train_df, test_df, tokenizer, batch_size=8, max_token_len=128):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def setup(self, stage=None):
        self.train_dataset = CommentsDataset(
          self.train_df,
          self.tokenizer,
          self.max_token_len
        )
        self.test_dataset = CommentsDataset(
          self.test_df,
          self.tokenizer,
          self.max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
          self.train_dataset,
          batch_size=self.batch_size,
          shuffle=True,
          num_workers=2
        )

    def val_dataloader(self):
        return DataLoader(
          self.test_dataset,
          batch_size=self.batch_size,
          num_workers=2
        )

    def test_dataloader(self):
        return DataLoader(
          self.test_dataset,
          batch_size=self.batch_size,
          num_workers=2
        )