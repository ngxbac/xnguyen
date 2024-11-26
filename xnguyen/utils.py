import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

from collections import defaultdict, deque
from typing import List, Optional
from abc import ABC, abstractmethod
import torch.nn as nn
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


def torch_to_Image(x):
    if x.ndim == 4:
        x = x[0]
    return transforms.ToPILImage()(x)


def decode_latents(latents, vae):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    return image


def batchwise_cosine_similarity(Z, B):
    # https://www.h4pz.co/blog/2021/4/2/batch-cosine-similarity-in-pytorch-or-numpy-jax-cupy-etc
    B = B.T
    Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True)  # Size (n, 1).
    B_norm = torch.linalg.norm(B, dim=0, keepdim=True)  # Size (1, b).
    cosine_similarity = ((Z @ B) / (Z_norm @ B_norm)).T
    return cosine_similarity


@torch.no_grad()
def reconstruction(
    batch,
    # image,
    # fmri_2d,
    voxel2clip,
    clip_extractor,
    unet,
    vae,
    noise_scheduler,
    img_lowlevel=None,
    num_inference_steps=50,
    recons_per_sample=1,
    guidance_scale=7.5,
    img2img_strength=0.85,
    seed=42,
    plotting=True,
    verbose=False,
    n_samples_save=1,
    device=None,
    mem_efficient=True,
):
    assert (
        n_samples_save == 1
    ), "n_samples_save must = 1. Function must be called one image at a time"
    assert recons_per_sample > 0, "recons_per_sample must > 0"

    brain_recons = None

    # fmri_2d = fmri_2d[:n_samples_save]
    # batch["fmri_2d"] = torch.rand_like(batch["fmri_2d"])
    image = batch["image"]
    caption = batch["caption"][0]
    image = image[:n_samples_save]
    B = image.shape[0]

    if mem_efficient:
        clip_extractor.to("cpu")
        unet.to("cpu")
        vae.to("cpu")
    else:
        clip_extractor.to(device)
        unet.to(device)
        vae.to(device)

    if unet is not None:
        do_classifier_free_guidance = guidance_scale > 1.0
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        height = unet.config.sample_size * vae_scale_factor
        width = unet.config.sample_size * vae_scale_factor
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    if voxel2clip is not None:
        clip_results = voxel2clip(batch, inference=True)
        # print(clip_results[0])

        if mem_efficient:
            voxel2clip.to("cpu")

        brain_clip_image_embeddings, brain_clip_text_embeddings = clip_results[:2]
        brain_clip_image_embeddings = brain_clip_image_embeddings.reshape(B, -1, 768)
        brain_clip_text_embeddings = brain_clip_text_embeddings.reshape(B, -1, 768)

        brain_clip_image_embeddings = brain_clip_image_embeddings.repeat(
            recons_per_sample, 1, 1
        )
        brain_clip_text_embeddings = brain_clip_text_embeddings.repeat(
            recons_per_sample, 1, 1
        )

    if recons_per_sample > 0:
        for samp in range(len(brain_clip_image_embeddings)):
            brain_clip_image_embeddings[samp] = brain_clip_image_embeddings[samp] / (
                brain_clip_image_embeddings[samp, 0].norm(dim=-1).reshape(-1, 1, 1)
                + 1e-6
            )
            brain_clip_text_embeddings[samp] = brain_clip_text_embeddings[samp] / (
                brain_clip_text_embeddings[samp, 0].norm(dim=-1).reshape(-1, 1, 1)
                + 1e-6
            )
        input_embedding = (
            brain_clip_image_embeddings  # .repeat(recons_per_sample, 1, 1)
        )
        if verbose:
            print("input_embedding", input_embedding.shape)

        prompt_embeds = brain_clip_text_embeddings
        if verbose:
            print("prompt_embedding", prompt_embeds.shape)

        if do_classifier_free_guidance:
            input_embedding = (
                torch.cat([torch.zeros_like(input_embedding), input_embedding])
                .to(device)
                .to(unet.dtype)
            )
            prompt_embeds = (
                torch.cat([torch.zeros_like(prompt_embeds), prompt_embeds])
                .to(device)
                .to(unet.dtype)
            )

        # 3. dual_prompt_embeddings
        input_embedding = torch.cat([prompt_embeds, input_embedding], dim=1)

        # 4. Prepare timesteps
        noise_scheduler.set_timesteps(
            num_inference_steps=num_inference_steps, device=device
        )

        # 5b. Prepare latent variables
        batch_size = (
            input_embedding.shape[0] // 2
        )  # divide by 2 bc we doubled it for classifier-free guidance
        shape = (
            batch_size,
            unet.in_channels,
            height // vae_scale_factor,
            width // vae_scale_factor,
        )
        if img_lowlevel is not None:  # use img_lowlevel for img2img initialization
            init_timestep = min(
                int(num_inference_steps * img2img_strength), num_inference_steps
            )
            t_start = max(num_inference_steps - init_timestep, 0)
            timesteps = noise_scheduler.timesteps[t_start:]
            latent_timestep = timesteps[:1].repeat(batch_size)

            if verbose:
                print("img_lowlevel", img_lowlevel.shape)
            img_lowlevel_embeddings = clip_extractor.normalize(img_lowlevel)
            if verbose:
                print("img_lowlevel_embeddings", img_lowlevel_embeddings.shape)
            if mem_efficient:
                vae.to(device)
            init_latents = vae.encode(
                img_lowlevel_embeddings.to(device).to(vae.dtype)
            ).latent_dist.sample(generator)
            init_latents = vae.config.scaling_factor * init_latents
            init_latents = init_latents.repeat(recons_per_sample, 1, 1, 1)

            noise = torch.randn(
                [recons_per_sample, 4, 64, 64],
                device=device,
                generator=generator,
                dtype=input_embedding.dtype,
            )
            init_latents = noise_scheduler.add_noise(
                init_latents, noise, latent_timestep
            )
            latents = init_latents
        else:
            timesteps = noise_scheduler.timesteps
            latents = torch.randn(
                [recons_per_sample, 4, 64, 64],
                device=device,
                generator=generator,
                dtype=input_embedding.dtype,
            )
            latents = latents * noise_scheduler.init_noise_sigma

        # 7. Denoising loop
        if mem_efficient:
            unet.to(device)
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = noise_scheduler.scale_model_input(
                latent_model_input, t
            )
            if verbose:
                print(
                    "timesteps: {}, latent_model_input: {}, input_embedding: {}".format(
                        i, latent_model_input.shape, input_embedding.shape
                    )
                )
            noise_pred = unet(
                latent_model_input, t, encoder_hidden_states=input_embedding
            ).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # compute the previous noisy sample x_t -> x_t-1
            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

        if mem_efficient:
            unet.to("cpu")

        recons = decode_latents(latents.to(device), vae.to(device)).detach().cpu()

        brain_recons = recons.unsqueeze(0)

    if verbose:
        print("brain_recons", brain_recons.shape)

    # pick best reconstruction out of several
    best_picks = np.zeros(n_samples_save).astype(np.int16)

    if mem_efficient:
        vae.to("cpu")
        unet.to("cpu")
        clip_extractor.to(device)

    clip_image_target = clip_extractor.embed_image(image)
    clip_image_target_norm = nn.functional.normalize(
        clip_image_target.flatten(1), dim=-1
    )
    sims = []
    for im in range(recons_per_sample):
        currecon = (
            clip_extractor.embed_image(brain_recons[0, [im]].float())
            .to(clip_image_target_norm.device)
            .to(clip_image_target_norm.dtype)
        )
        currecon = nn.functional.normalize(currecon.view(len(currecon), -1), dim=-1)
        cursim = batchwise_cosine_similarity(clip_image_target_norm, currecon)
        sims.append(cursim.item())
    if verbose:
        print(sims)
    best_picks[0] = int(np.nanargmax(sims))
    if verbose:
        print(best_picks)
    if mem_efficient:
        clip_extractor.to("cpu")
        voxel2clip.to(device)

    img2img_samples = 0 if img_lowlevel is None else 1
    num_xaxis_subplots = 1 + img2img_samples + recons_per_sample
    if plotting:
        fig, ax = plt.subplots(
            n_samples_save,
            num_xaxis_subplots,
            figsize=(num_xaxis_subplots * 5, 6 * n_samples_save),
            facecolor=(1, 1, 1),
        )
    else:
        fig = None
        recon_img = None

    im = 0
    if plotting:
        ax[0].set_title(f"Original Image: \n{caption}")
        ax[0].imshow(torch_to_Image(image[im]))
        if img2img_samples == 1:
            ax[1].set_title(f"Img2img ({img2img_strength})")
            ax[1].imshow(torch_to_Image(img_lowlevel[im].clamp(0, 1)))
    for ii, i in enumerate(
        range(num_xaxis_subplots - recons_per_sample, num_xaxis_subplots)
    ):
        recon = brain_recons[im][ii]
        if plotting:
            if ii == best_picks[im]:
                ax[i].set_title(f"Reconstruction", fontweight="bold")
                recon_img = recon
            else:
                ax[i].set_title(f"Recon {ii+1} from brain")
            ax[i].imshow(torch_to_Image(recon))
    if plotting:
        for i in range(num_xaxis_subplots):
            ax[i].axis("off")

    return fig, brain_recons, best_picks, recon_img


def pairwise_cosine_similarity(A, B, dim=1, eps=1e-8):
    # https://stackoverflow.com/questions/67199317/pytorch-cosine-similarity-nxn-elements
    numerator = A @ B.T
    A_l2 = torch.mul(A, A).sum(axis=dim)
    B_l2 = torch.mul(B, B).sum(axis=dim)
    denominator = torch.max(torch.sqrt(torch.outer(A_l2, B_l2)), torch.tensor(eps))
    return torch.div(numerator, denominator)


def batchwise_cosine_similarity(Z, B):
    # https://www.h4pz.co/blog/2021/4/2/batch-cosine-similarity-in-pytorch-or-numpy-jax-cupy-etc
    B = B.T
    Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True)  # Size (n, 1).
    B_norm = torch.linalg.norm(B, dim=0, keepdim=True)  # Size (1, b).
    cosine_similarity = ((Z @ B) / (Z_norm @ B_norm)).T
    return cosine_similarity


def topk(similarities, labels, k=5):
    if k > similarities.shape[0]:
        k = similarities.shape[0]
    topsum = 0
    for i in range(k):
        topsum += torch.sum(
            torch.argsort(similarities, axis=1)[:, -(i + 1)] == labels
        ) / len(labels)
    return topsum
