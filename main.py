import torch
import torch.nn as nn
from trainer_base import AcceleratorTrainer
from data import get_dls
from options import args
import os
from utils import f1_score


class Criterion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.l1 = nn.SmoothL1Loss()

    def subject_common_loss(self, output_dict, batch):
        sc_logit = output_dict["sc_logit"]
        instances = batch["instance"]
        # loss_sc = self.bce(sc_logit, instances)
        loss_sc = 1 - f1_score(sc_logit.sigmoid(), instances)
        return loss_sc

    def subject_specific_loss(self, output_dict, batch):
        ss_logit = output_dict["ss_logit"]
        subject = batch["subject"]
        loss_ss = self.ce(ss_logit, subject)
        return loss_ss

    def soft_clip_loss(self, preds, targs, temp=0.005, eps=1e-10):
        def check_loss(loss, message="loss"):
            if loss.isnan().any():
                raise ValueError(f"NaN loss in {message}")

        clip_clip = (targs @ targs.T) / temp + eps
        check_loss(clip_clip, "clip_clip")
        brain_clip = (preds @ targs.T) / temp + eps
        check_loss(brain_clip, "brain_clip")

        loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        check_loss(loss1, "loss1")
        loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        check_loss(loss2, "loss2")

        loss = (loss1 + loss2) / 2
        return loss

    # def clip_loss(self, pred, target):
    #     pred_norm = nn.functional.normalize(pred.flatten(1), dim=-1)
    #     target_norm = nn.functional.normalize(target.flatten(1), dim=-1)
    #     clip_loss = self.soft_clip_loss(pred_norm, target_norm)

    #     mse_loss = self.mse(pred_norm, target_norm)
    #     l1_loss = self.l1(pred_norm, target_norm)
    #     return clip_loss, mse_loss, l1_loss

    # contrastive loss function, adapted from
    # https://sachinruk.github.io/blog/2021-03-07-clip.html
    def contrastive_loss(self, logits) -> torch.Tensor:
        return nn.functional.cross_entropy(
            logits,
            torch.arange(len(logits), device=logits.device),
        )

    def self_clip_loss(self, output_dict, batch):
        x = output_dict["first_cls_token"]
        scale = output_dict["logit_scale"]
        x = nn.functional.normalize(x.flatten(1), dim=-1)
        similarity = x @ x.T
        similarity = similarity * scale.exp()
        caption_loss = self.contrastive_loss(similarity)  # .mean()
        image_loss = self.contrastive_loss(similarity.t())  # .mean()
        loss = (caption_loss + image_loss) / 2.0
        return loss

    def l1_loss(self, pred, gt, norm=True):
        if norm:
            pred = nn.functional.normalize(pred, dim=-1)
            gt = nn.functional.normalize(gt, dim=-1)

        return self.l1(pred, gt)

    def reconstruction_loss(self, output_dict, batch):
        clip_image_pred = output_dict["clip_image"]
        clip_text_pred = output_dict["clip_text"]

        clip_image = batch["clip_image"]
        clip_text = batch["clip_text"]

        l1_image = self.l1_loss(clip_image_pred, clip_image)
        l1_text = self.l1_loss(clip_text_pred, clip_text)

        return l1_image + l1_text

    def coco_id_loss(self, output_dict, batch):
        pred = output_dict["coco_logit"]
        gt = batch["coco_id"]
        return self.ce(pred, gt)

    def forward(self, output_dict, batch):
        loss_sc = self.subject_common_loss(output_dict, batch)
        loss_ss = self.subject_specific_loss(output_dict, batch)
        loss_recon = self.reconstruction_loss(output_dict, batch)
        loss_self_clip = self.self_clip_loss(output_dict, batch)
        loss_coco_id = self.coco_id_loss(output_dict, batch)

        loss = loss_sc + loss_ss + loss_recon * 144 + loss_self_clip + loss_coco_id
        ret_dict = {
            "loss": loss,
            "metrics": {
                "loss_sc": loss_sc,
                "loss_ss": loss_ss,
                "loss_self_clip": loss_self_clip,
                "loss_recon": loss_recon,
                "loss_coco_id": loss_coco_id,
            },
        }

        return ret_dict


class OpenMindTrainer(AcceleratorTrainer):
    def __init__(self, args):
        super().__init__(args)

    def pre_init(self):
        self.clip_extractor, self.output_dim_image, self.output_dim_text = (
            self.prepare_CLIP()
        )
        self.clip_extractor.eval()

        import kornia
        from kornia.augmentation.container import AugmentationSequential

        self.img_augment = AugmentationSequential(
            kornia.augmentation.RandomResizedCrop((224, 224), (0.6, 1), p=0.3),
            kornia.augmentation.Resize((224, 224)),
            kornia.augmentation.RandomHorizontalFlip(p=0.5),
            kornia.augmentation.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.3
            ),
            kornia.augmentation.RandomGrayscale(p=0.3),
            data_keys=["input"],
        )

    def get_criterion(self):
        return Criterion(self.args)

    def get_dataloader(self):
        return get_dls(
            subjects=self.args.subject,
            root_dir=self.args.data_path,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
        )

    def get_model(self):
        from models.openmind import OpenMind

        model = OpenMind(self.args)

        if os.path.isfile(args.cl_checkpoint):
            state_dict = torch.load(args.cl_checkpoint, map_location="cpu")["model"]
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[k.replace("module.", "")] = v

            mes = model.load_state_dict(new_state_dict, strict=False)
            print(f"Contiual learning from: {args.cl_checkpoint}")
            print(mes)
            model.freeze_for_cl()
        return model

    def mixco(self, voxels, beta=0.15, s_thresh=0.5):
        perm = torch.randperm(voxels.shape[0])
        voxels_shuffle = voxels[perm].to(voxels.device, dtype=voxels.dtype)
        betas = (
            torch.distributions.Beta(beta, beta)
            .sample([voxels.shape[0]])
            .to(voxels.device, dtype=voxels.dtype)
        )
        select = (torch.rand(voxels.shape[0]) <= s_thresh).to(voxels.device)
        betas_shape = [-1] + [1] * (len(voxels.shape) - 1)
        voxels[select] = voxels[select] * betas[select].reshape(
            *betas_shape
        ) + voxels_shuffle[select] * (1 - betas[select]).reshape(*betas_shape)
        betas[~select] = 1
        return voxels, perm, betas, select

    def mixco_clip_target(self, clip_target, perm, select, betas):
        clip_target_shuffle = clip_target[perm]
        clip_target[select] = clip_target[select] * betas[select].reshape(
            -1, 1
        ) + clip_target_shuffle[select] * (1 - betas[select]).reshape(-1, 1)
        return clip_target

    def pre_forward(self, batch):
        image = batch["image"]
        image = self.img_augment(image)

        use_mixup = (
            self.epoch < int(self.args.mixup_pct * self.args.epochs)
            and self.model.training
        )

        batch["use_mixup"] = use_mixup

        if use_mixup:
            fmri_2d = batch["fmri_2d"]
            fmri_2d, perm, betas, select = self.mixco(fmri_2d)
            batch["perm"] = perm
            batch["betas"] = betas
            batch["select"] = select
            batch["fmri_2d"] = fmri_2d

        with torch.no_grad():
            clip_image = self.clip_extractor.embed_image(image).float()
            clip_text = self.clip_extractor.embed_text(batch["caption"]).float()
            batch["clip_image"] = clip_image.detach()
            batch["clip_text"] = clip_text.detach()

        return batch

    def pos_forward(self, loss, is_train=True):
        import math
        import sys

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        if not is_train:
            return

        self.accelerator.backward(loss)
        self.optimizer.step()
        self.scheduler.step()

    def save(self, epoch, is_best):
        if (epoch + 1) % self.args.saveckp_freq == 0:
            super().save(epoch, is_best)

    def prepare_CLIP(self):
        from models.clipper import Clipper

        # Prepare CLIP
        clip_sizes = {"RN50": 1024, "ViT-L/14": 768, "ViT-B/32": 512, "ViT-H-14": 1024}
        clip_size = clip_sizes[self.args.clip_variant]

        print("Using hidden layer CLIP space (Versatile Diffusion)")
        if not self.args.norm_embs:
            print("WARNING: YOU WANT NORMED EMBEDDINGS FOR VERSATILE DIFFUSION!")
        clip_extractor = Clipper(
            args.clip_variant,
            device=self.accelerator.device,
            hidden_state=True,
            norm_embs=self.args.norm_embs,
        )

        out_dim_image = 257 * clip_size  # 257*768 = 197376
        out_dim_text = 77 * clip_size  # 77*768  = 59136

        print("clip_extractor loaded.")
        print("out_dim_image:", out_dim_image)
        print("out_dim_text:", out_dim_text)

        return clip_extractor, out_dim_image, out_dim_text


if __name__ == "__main__":
    print(args)
    trainer = OpenMindTrainer(args=args)
    trainer.train()
