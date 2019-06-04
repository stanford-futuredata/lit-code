import torch
import torch.cuda
import torch.nn as nn
from collections import OrderedDict
import threading


def get_a_var(obj):
    if isinstance(obj, torch.Tensor):
        return obj

    if isinstance(obj, list) or isinstance(obj, tuple):
        for result in map(get_a_var, obj):
            if isinstance(result, torch.Tensor):
                return result
    if isinstance(obj, dict):
        for result in map(get_a_var, obj.items()):
            if isinstance(result, torch.Tensor):
                return result
    return None


def parallel_apply(modules, inputs, kwargs_tup=None, devices=None):
    assert len(modules) == len(inputs)
    if kwargs_tup is not None:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)

    lock = threading.Lock()
    results = {}
    grad_enabled = torch.is_grad_enabled()

    def _worker(i, module, input, kwargs, device=None):
        torch.set_grad_enabled(grad_enabled)
        if device is None:
            device = get_a_var(input).get_device()
        try:
            with torch.cuda.device(device):
                # this also avoids accidental slicing of `input` if it is a Tensor
                if not isinstance(input, (list, tuple)):
                    input = (input,)
                output = module(*input, **kwargs)
            with lock:
                results[i] = output
        except Exception as e:
            with lock:
                results[i] = e

    if len(modules) > 1:
        threads = [threading.Thread(target=_worker,
                                    args=(i, module, input, kwargs, device))
                   for i, (module, input, kwargs, device) in
                   enumerate(zip(modules, inputs, kwargs_tup, devices))]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, Exception):
            raise output
        outputs.append(output)
    return outputs

class LearnerModelParallel(nn.Module):
    def __init__(self, module, sections):
        super(LearnerModelParallel, self).__init__()

        self.module = module.cuda()
        self.sections = sections
        self.num_sections = len(self.sections)

        # Distribute sections to GPUs by number of trainable parameters
        self._scatter_sections()

    def _scatter_sections(self):
        # We use number of trainable parameters as a proxy for computational & memory burden on GPU
        num_parameters = lambda s: sum(p.numel() for p in s.parameters() if p.requires_grad)
        gpu_ids = list(range(torch.cuda.device_count()))
        available_devices = [i for i in gpu_ids]
        sorted_sections = sorted(self.sections,
                                    key= lambda x: num_parameters(self.sections[x].network),
                                    reverse=True)

        # Greedily allocate the section to the most available gpu
        device_load = dict((i, 0) for i in available_devices)
        for id in sorted_sections:
            device = min(device_load, key=device_load.get)
            device_load[device] += num_parameters(self.sections[id].network)
            device = torch.device("cuda:{}".format(device))
            self.sections[id].network.to(device)
            self.sections[id].device = device

    def cpu(self):
        for i, s in self.sections.items():
            self.sections[i].network = s.network.cpu()
            self.sections[i].device = torch.device("cpu")
        self.module = self.module.cpu()

    def _scatter_features(self, features):
        for id in self.sections:
            inp = features[id-1]
            features[id-1] = inp.detach().cuda(self.sections[id].device, non_blocking = True)

    def forward(self, features):
        self._scatter_features(features)
        # Now call forward for each of the sections
        modules = self.sections.values()
        inputs = list(features.values())[:-1]
        outputs = parallel_apply(modules, inputs)
        outputs = OrderedDict((i+1, outputs[i]) for i in range(len(outputs)))
        return outputs
