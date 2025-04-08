from torch.utils.data import Dataset, DataLoader
import torch


class DatasetForCausalLM(Dataset):
    def __init__(self, chunks):
        self.chunks = torch.tensor(chunks)
        
    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        inputs = self.chunks[idx][:-1].long()
        targets = self.chunks[idx][1:].long()
        return inputs, targets


def load_data(ds: Dataset, batch_size: int) -> DataLoader:
	return DataLoader(
		ds,
		batch_size=batch_size,
		shuffle=True,
		drop_last=True
	)