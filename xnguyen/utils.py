from collections import defaultdict, deque
import torch
import time
import datetime
import torch.distributed as dist


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        # if not is_dist_avail_and_initialized():
        #     return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.6f}")
        data_time = SmoothedValue(fmt="{avg:.6f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.6f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100.0 / batch_size for k in topk]


def f1_score(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    beta: float = 1.0,
    eps: float = 1e-7,
    threshold: float = None,
    activation=None,
):
    """
    Args:
        outputs (torch.Tensor): A list of predicted elements
        targets (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        beta (float): beta param for f_score
        threshold (float): threshold for outputs binarization
    Returns:
        float: F_1 score
    """
    if activation is not None:
        activation_fn = F.sigmoid
    else:
        activation_fn = torch.nn.Identity()

    outputs = activation_fn(outputs)

    if threshold is not None:
        outputs = (outputs > threshold).float()

    true_positive = torch.sum(targets * outputs)
    false_positive = torch.sum(outputs) - true_positive
    false_negative = torch.sum(targets) - true_positive

    precision_plus_recall = (
        (1 + beta**2) * true_positive + beta**2 * false_negative + false_positive + eps
    )

    score = ((1 + beta**2) * true_positive + eps) / precision_plus_recall

    return score


def topk(similarities, labels, k=5):
    if k > similarities.shape[0]:
        k = similarities.shape[0]
    topsum = 0
    for i in range(k):
        topsum += torch.sum(
            torch.argsort(similarities, axis=1)[:, -(i + 1)] == labels
        ) / len(labels)
    return topsum


import torch

def load_pretrained(model, pretrained_path):
    checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    checkpoint = checkpoint["model"] if "model" in checkpoint else checkpoint

    # Clean up keys: remove .module prefix
    new_state_dict = {}
    for k, v in checkpoint.items():
        kk = k.replace(".module", "")  # handle DDP
        new_state_dict[kk] = v

    # Reference model parameters
    target_model = model.module if hasattr(model, "module") else model
    model_state = target_model.state_dict()

    # Filter for matching shape
    compatible_state_dict = {}
    mismatched_keys = []

    for k, v in new_state_dict.items():
        if k in model_state:
            if v.shape == model_state[k].shape:
                compatible_state_dict[k] = v
            else:
                mismatched_keys.append((k, v.shape, model_state[k].shape))
        else:
            # Still include; load_state_dict with strict=False will ignore unexpected
            compatible_state_dict[k] = v

    # Load filtered state dict
    missing, unexpected = target_model.load_state_dict(compatible_state_dict, strict=False)

    # Report
    if missing:
        print(f"[Warning] Missing keys: {missing}")
    if unexpected:
        print(f"[Warning] Unexpected keys: {unexpected}")
    if mismatched_keys:
        print("[Warning] Mismatched keys:")
        for k, ckpt_shape, model_shape in mismatched_keys:
            print(f" - {k}: checkpoint shape {ckpt_shape}, model shape {model_shape}")

    return model

