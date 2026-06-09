import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}
    for key in dataset_items[0].keys():
        values = [item[key] for item in dataset_items]
        if torch.is_tensor(values[0]):
            result_batch[key] = torch.stack(values)
        else:
            result_batch[key] = values

    return result_batch
