"""
LIDC-IDRI U-Net 肺结节分割项目 (PyTorch 版本)
"""
from .dataset import LIDCDataset, create_dataloaders
from .model import UNet, UNetSmall, UNetLarge, build_unet
from .losses import (
    DiceLoss, IoULoss, BCEDiceLoss, FocalLoss, TverskyLoss,
    dice_coef, iou_score, get_loss_function
)
from .metrics import (
    dice_coefficient, iou_score as iou_metric, precision, recall,
    compute_all_metrics, evaluate_model, MetricTracker
)
from .utils import (
    plot_training_history, visualize_predictions,
    TrainingLogger, EarlyStopping, print_gpu_info,
    save_checkpoint, load_checkpoint
)
