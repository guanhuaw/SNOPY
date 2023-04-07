import importlib
import torch.utils.data
from data.base_data_loader import BaseDataLoader
from data.base_dataset import BaseDataset
from torch.utils.data.sampler import Sampler
import random

def find_dataset_using_name(dataset_name):
    # Given the option --dataset_mode [datasetname],
    # the file "data/datasetname_dataset.py"
    # will be imported.
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    # In the file, the class called DatasetNameDataset() will
    # be instantiated. It has to be a subclass of BaseDataset,
    # and it is case-insensitive.
    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls
            
    if dataset is None:
        print("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))
        exit(0)

    return dataset


def get_option_setter(dataset_name):    
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    dataset = find_dataset_using_name(opt.dataset_mode)
    instance = dataset()
    instance.initialize(opt)
    print("dataset [%s] was created" % (instance.name()))
    return instance


def CreateDataLoader(opt):
    data_loader = CustomDatasetDataLoader()
    data_loader.initialize(opt)
    return data_loader

class ClusterRandomSampler(Sampler):
    r"""Takes a dataset with cluster_indices property, cuts it into batch-sized chunks
    Drops the extra items, not fitting into exact batches
    Arguments:
        data_source (Dataset): a Dataset to sample from. Should have a cluster_indices property
        batch_size (int): a batch size that you would like to use later with Dataloader class
        shuffle (bool): whether to shuffle the data or not
    """

    def __init__(self, data_source, batch_size, shuffle=False):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        batch_lists = []
        for cluster_indices in self.data_source.cluster_indices:
            if cluster_indices is not []:
                # random.shuffle(cluster_indices)
                batches = [cluster_indices[i:i + self.batch_size] for i in range(0, len(cluster_indices), self.batch_size)]
                # filter our the shorter batches
                batches = [_ for _ in batches if len(_) == self.batch_size]
                if self.shuffle:
                    random.seed(0)
                    random.shuffle(batches)
                batch_lists.append(batches)
            # flatten lists and shuffle the batches if necessary
        # this works on batch level
        lst = self.flatten_list(batch_lists)
        if self.shuffle:
            random.seed(0)
            random.shuffle(lst)
        # final flatten  - produce flat list of indexes
        self.lst = self.flatten_list(lst)
        print('len real', len(self.lst))
        print('len source', len(self.data_source))

    def flatten_list(self, lst):
        return [item for sublist in lst for item in sublist]

    def __iter__(self):


        return iter(self.lst)


    def __len__(self):
        return len(self.lst)

## Wrapper class of Dataset class that performs
## multi-threaded data loading
class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = create_dataset(opt)
        if opt.dataset_mode != 'fastmri':
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batchSize,
                shuffle=not opt.serial_batches,
                num_workers=int(opt.nThreads))
        else:
            self.customsampler = ClusterRandomSampler(self.dataset, opt.batchSize, shuffle=not opt.serial_batches)
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                sampler=self.customsampler,
                batch_size=opt.batchSize,
                num_workers=int(opt.nThreads),
                drop_last=False,
                shuffle=False)


    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batchSize >= self.opt.max_dataset_size:
                break
            yield data
