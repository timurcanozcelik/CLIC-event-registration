import os
import shutil
import logging
from argparse import ArgumentParser
import pandas as pd
import torch
from torch import nn

# Lightning is needed only because the checkpoint was saved with PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.utilities.parsing import AttributeDict
from AuDrA_DataModule import user_Dataloader
from AuDrA_Class import AuDrA

# ---------- Logging setup ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AuDrA_run")

# ---------- CLI ----------
parser = ArgumentParser(description="Run AuDrA on given drawings.")
parser.add_argument(
    "--output-filename",
    default="AuDrA_predictions.csv",
    help="Name/path of the output CSV with AuDrA predictions",
)
parser.add_argument(
    "--input-dir",
    default="user_images",
    help="Directory containing user drawings (will be copied into 'user_images' for compatibility)",
)
parser.add_argument("--architecture", default="resnet18")
parser.add_argument(
    "--pretrained",
    action="store_true",
    help="Use pretrained weights for backbone (default behavior unless --no-pretrained is given)",
)
parser.add_argument(
    "--no-pretrained",
    dest="pretrained",
    action="store_false",
    help="Do not use pretrained weights",
)
parser.set_defaults(pretrained=True)
parser.add_argument(
    "--in_shape",
    nargs=3,
    type=int,
    default=[3, 224, 224],
    help="Input shape as three ints",
)
parser.add_argument("--img_means", default=0.1612, type=float)
parser.add_argument("--img_stds", default=0.4075, type=float)
parser.add_argument("--num_outputs", default=1, type=int)
parser.add_argument("--learning_rate", default=0.00034664640432471026, type=float)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--train_pct", default=0.7, type=float)
parser.add_argument("--val_pct", default=0.1, type=float)
parser.add_argument(
    "--loss_func", default=nn.MSELoss(), type=object, help="Loss function object"
)
parser.add_argument("--num_workers", default=1, type=int)
args = parser.parse_args()

# Normalize some args into expected shapes
args.in_shape = list(args.in_shape)  # ensure list for downstream

# Prepare output path
output_filename = args.output_filename
output_dir = os.path.dirname(output_filename)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# ---------- Prepare user_images compatibility ----------
input_dir_raw = args.input_dir
if not os.path.isabs(input_dir_raw):
    input_dir = os.path.abspath(input_dir_raw)
else:
    input_dir = input_dir_raw

if not os.path.isdir(input_dir):
    logger.error("Input directory '%s' does not exist (resolved to '%s')", args.input_dir, input_dir)
    raise FileNotFoundError(f"Input directory '{input_dir}' not found.")

# Ensure clean user_images as expected by existing dataloader
if os.path.exists("user_images"):
    shutil.rmtree("user_images")
shutil.copytree(input_dir, "user_images")
logger.info("Copied '%s' into local 'user_images' for compatibility.", input_dir)

# ---------- DEVICE ----------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logger.info("Running AuDrA on %s", "GPU" if device.type == "cuda" else "CPU")

# ---------- LOAD MODEL ----------
model = AuDrA(args)

ckpt_path = "AuDrA_trained.ckpt"
if not os.path.exists(ckpt_path):
    logger.error("Checkpoint file not found at '%s'", ckpt_path)
    raise FileNotFoundError(f"Checkpoint '{ckpt_path}' does not exist.")

# Safe load checkpoint, allowing the Lightning AttributeDict global if needed
model_weights = None
try:
    with torch.serialization.safe_globals([AttributeDict]):
        model_weights = torch.load(ckpt_path, map_location=device)
    logger.info("Loaded checkpoint with safe_globals.")
except Exception as e:
    logger.warning("Loading with safe_globals failed (%s); trying with weights_only=False fallback.", e)
    try:
        model_weights = torch.load(ckpt_path, map_location=device, weights_only=False)
        logger.info("Loaded checkpoint with fallback weights_only=False.")
    except Exception as e2:
        logger.error("Failed to load checkpoint on fallback: %s", e2)
        raise

# Load state dict into model
if isinstance(model_weights, dict) and "state_dict" in model_weights:
    model.load_state_dict(model_weights["state_dict"])
else:
    model.load_state_dict(model_weights)
model.eval()
model.to(device)

# ---------- LOAD IMAGES AND GET PREDICTIONS ----------
dataloader = user_Dataloader(args=args)

filenames = []
predictions = []

for idx, img in enumerate(dataloader):
    try:
        fname = img[0][0]
        x = img[1].to(device)
        # Use the model to predict; direct .forward is acceptable here given legacy
        pred = model.forward(x).item()
        filenames.append(fname)
        predictions.append(pred)
        logger.info("Predicted %s : %s", fname, pred)
    except Exception as e:
        logger.error("Error processing batch %s: %s", idx, e)

# ---------- SAVE OUTPUT ----------
out_df = pd.DataFrame(zip(filenames, predictions), columns=["filenames", "predictions"])
try:
    out_df.to_csv(output_filename, index=False)
    logger.info("Saved predictions to %s", output_filename)
except Exception as e:
    logger.error("Failed to write output CSV '%s': %s", output_filename, e)
    raise

# Also echo to stdout for legacy callers
print(f"Saved predictions to {output_filename}")
for fn, p in zip(filenames, predictions):
    print(f"{fn}: {p}")
