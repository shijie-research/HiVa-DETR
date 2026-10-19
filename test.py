"""
Evaluate a HiVa-DETR checkpoint on the validation or test split its config names.

    python test.py -c configs/HiVa-DETR-S-VisDrone.yml -r weights/hiva_detr_s_visdrone.pth
"""

import argparse
import time

import torch

from src.core import YAMLConfig, yaml_utils


def prediction_scale(target):
    """
    The (w, h) that maps a prediction normalized to the network input back to the original image:
    ``orig_size``, enlarged by ``padded_size / resized_size`` when the collate function padded it.
    """
    if "padded_size" in target:
        return target["orig_size"] * target["padded_size"] / target["resized_size"]
    return target["orig_size"]


@torch.no_grad()
def evaluate(model, postprocessor, data_loader, evaluator, device, print_freq=50):
    model.eval()
    evaluator.cleanup()
    queries = []
    start = time.time()
    for i, (samples, targets) in enumerate(data_loader):
        outputs = model(samples.to(device))
        sizes = torch.stack([prediction_scale(t) for t in targets], dim=0).to(device)
        results = postprocessor(outputs, sizes)
        evaluator.update(
            {int(t["image_id"]): {k: v.cpu() for k, v in r.items()} for t, r in zip(targets, results)}
        )
        queries.extend(outputs["batch_queries_num"])
        if i % print_freq == 0 or i == len(data_loader) - 1:
            print(f"[{i + 1}/{len(data_loader)}] {time.time() - start:.0f} s")

    print(f"queries per image: mean {sum(queries) / len(queries):.1f}, min {min(queries)}, max {max(queries)}")
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()
    stats = evaluator.coco_eval["bbox"].stats
    print("stats (%):", " ".join(f"{100 * s:.2f}" for s in stats))
    return evaluator


def main(args):
    update = yaml_utils.parse_cli(args.update)
    cfg = YAMLConfig(args.config, **update)
    device = torch.device(args.device)

    model = cfg.model
    state = torch.load(args.resume, map_location="cpu")
    model.load_state_dict(state["model"])
    # frozen as the EMA copy the reported results were evaluated with: nn.MultiheadAttention takes
    # its fused inference path only when no parameter requires a gradient
    model.requires_grad_(False)
    model.to(device)
    postprocessor = cfg.postprocessor.to(device)
    evaluate(model, postprocessor, cfg.val_dataloader, cfg.evaluator, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--resume", type=str, required=True, help="checkpoint to evaluate")
    parser.add_argument("-d", "--device", type=str, default="cuda")
    parser.add_argument("-u", "--update", nargs="+", help="override config entries, e.g. val_dataloader.num_workers=2")
    main(parser.parse_args())
