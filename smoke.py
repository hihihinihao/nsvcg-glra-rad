"""One-batch smoke test for GLRA front + RAD (CPU or GPU).

Example (MNRE):
  GLRA_N_MAIN=1 GLRA_N_AUX=2 NSVCG_BENCH=mnre \
  python smoke.py --data_path ~/mre/data/MNRE_GLRA
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from modules.dataset import MREProcessor, MNRE_MI_Dataset
from glra_front import GLRAFront
from train_glra_front_rad import GLRAFrontRAD


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--text_model_name", default="bert-base-uncased")
    p.add_argument("--image_model_name",
                   default=str(Path.home() / "mre/pretrained_models/ViTL-32-384"))
    p.add_argument("--max_seq_length", type=int, default=128)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    data = Path(args.data_path)
    args.image_path = str(data / "images")
    args.aux_image_path = str(data / "images_vg")
    args.aux_image_dict = str(data / "images_vg.pth")
    args.device = args.device
    processor = MREProcessor(args)
    ds = MNRE_MI_Dataset(processor, args, "train.txt")
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    front = GLRAFront(processor.tokenizer, args)
    model = GLRAFrontRAD(front, len(processor.get_relation_dict()))
    if args.device == "cuda":
        model.cuda()
        batch = [x.cuda() if torch.is_tensor(x) else x for x in batch]
    out = model(*batch[:5], batch[5])
    print("shapes", [tuple(x.shape) for x in out])
    loss = (torch.nn.functional.cross_entropy(out[0], batch[5])
            + 2 * torch.nn.functional.cross_entropy(out[1], batch[5]))
    loss.backward()
    print("loss", float(loss), "OK")


if __name__ == "__main__":
    main()
