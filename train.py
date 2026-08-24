import os
import argparse
import time
import datetime
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.utils as vutils
from util.MF_dataset import MF_dataset
from util.augmentation import RandomFlip, RandomCrop
from util.util import compute_results
from util.boundary_metrics import (
    calculate_boundary_metrics,
    create_boundary_stats,
    update_boundary_stats,
)
from sklearn.metrics import confusion_matrix
import swanlab
try:
    from .BSIFNet import BSIFNet
except ImportError:
    from BSIFNet import BSIFNet
from util.tversky import TverskyLoss
from util.soft_ce import SoftCrossEntropyLoss
from util.joint_loss import JointLoss
from PIL import Image

def _swanlab_log_bsifnet_code_once(step: int = 0):
    """将模型源码快照记录到 SwanLab。"""
    code_path = os.path.join(os.path.dirname(__file__), "BSIFNet.py")
    try:
        with open(code_path, "r", encoding="utf-8") as f:
            code = f.read()
        sha12 = hashlib.sha256(code.encode("utf-8")).hexdigest()[:12]
        st = os.stat(code_path)
        mtime = datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")
        swanlab.log(
            {
                "Code/BSIFNet.py": swanlab.Text(code),
                "Code/BSIFNet.py_meta": swanlab.Text(
                    f"path={code_path}\nbytes={st.st_size}\nmtime={mtime}\nsha256_12={sha12}"
                ),
            },
            step=step,
        )
    except Exception as e:
        print(f"[WARN] SwanLab 保存 BSIFNet.py 失败: {e}")

parser = argparse.ArgumentParser(description='Train with pytorch')
parser.add_argument('--model_name', '-m', type=str, default='BSIFNet')
parser.add_argument('--batch_size', '-b', type=int, default=5) 
parser.add_argument('--lr_start', '-ls', type=float, default=0.02)
parser.add_argument('--gpu', '-g', type=int, default=0)
parser.add_argument('--lr_decay', '-ld', type=float, default=0.95)
parser.add_argument('--epoch_max', '-em', type=int, default=10000)
parser.add_argument('--epoch_from', '-ef', type=int, default=0) 
parser.add_argument('--num_workers', '-j', type=int, default=8)
parser.add_argument('--n_class', '-nc', type=int, default=9)
parser.add_argument('--boundary_width', type=int, default=1,
                    help='Boundary band width in pixels for Boundary IoU')
parser.add_argument('--boundary_tolerance', type=int, default=2,
                    help='Pixel tolerance for matching boundaries in Boundary F-score')
parser.add_argument('--boundary_ignore_unlabeled', action='store_true',
                    help='Exclude class 0 (unlabeled) from boundary macro averages')
parser.add_argument('--data_dir', '-dr', type=str, default='./dataset/')
parser.add_argument('--boundary_dir', type=str, default='./processed_boundaries', help='边界监督图片目录')
parser.add_argument('--binary_dir', type=str, default='./processed_binary', help='二值监督图片目录')
parser.add_argument('--resume', '-r', type=str, default='', help='断点路径，例如 ./runs/BSIFNet/best.pth')
args = parser.parse_args()
if args.boundary_width < 1:
    parser.error('--boundary_width must be >= 1')
if args.boundary_tolerance < 0:
    parser.error('--boundary_tolerance must be >= 0')
augmentation_methods = [
    RandomFlip(prob=0.5),
    RandomCrop(crop_rate=0.1, prob=1.0),
]

criterion = JointLoss(
    first=TverskyLoss(mode='multiclass', alpha=0.7, beta=0.3),
    second=SoftCrossEntropyLoss(smooth_factor=0.1),
    first_weight=0.5,
    second_weight=0.5,
)

def _load_mask_as_long(path, target_size):
    """加载并缩放二值监督图；文件不存在时返回 None。"""
    try:
        img = np.asarray(Image.open(path))
        if img.ndim == 3:
            img = img[:, :, 0]
        img = np.asarray(Image.fromarray(img).resize(
            (target_size[1], target_size[0]), resample=Image.NEAREST))
        return torch.from_numpy((img > 0).astype(np.int64))
    except Exception:
        return None

def load_boundary_labels_batch(names, boundary_dir, target_size, device):
    """加载一个 batch 的边界监督图。"""
    H, W = target_size
    bnd_list = []
    bnd_valid = []
    for name in names:
        bnd_path = os.path.join(boundary_dir, name + '_boundary.png')
        t = _load_mask_as_long(bnd_path, (H, W))
        if t is not None:
            bnd_list.append(t); bnd_valid.append(True)
        else:
            bnd_list.append(torch.zeros(H, W, dtype=torch.int64)); bnd_valid.append(False)
    boundary_labels = torch.stack(bnd_list, dim=0).to(device)
    boundary_valid  = torch.tensor(bnd_valid, dtype=torch.bool, device=device)
    return boundary_labels, boundary_valid

def load_binary_labels_batch(names, binary_dir, target_size, device):
    """加载一个 batch 的二值监督图。"""
    H, W = target_size
    bin_list = []
    bin_valid = []
    for name in names:
        bin_path = os.path.join(binary_dir, name + '_binary.png')
        t = _load_mask_as_long(bin_path, (H, W))
        if t is not None:
            bin_list.append(t); bin_valid.append(True)
        else:
            bin_list.append(torch.zeros(H, W, dtype=torch.int64)); bin_valid.append(False)
    binary_labels = torch.stack(bin_list, dim=0).to(device)
    binary_valid  = torch.tensor(bin_valid, dtype=torch.bool, device=device)
    return binary_labels, binary_valid

def compute_edge_loss(logits, labels, valid_mask):
    """计算边界分支损失，仅使用有监督样本。"""
    if not valid_mask.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    logits_v = logits[valid_mask]
    labels_v = labels[valid_mask]
    if logits_v.shape[2] != labels_v.shape[1] or logits_v.shape[3] != labels_v.shape[2]:
        labels_v = F.interpolate(
            labels_v.unsqueeze(1).float(),
            size=(logits_v.shape[2], logits_v.shape[3]),
            mode='nearest'
        ).squeeze(1).long()
    weight = torch.tensor([1.0, 20.0], device=logits.device)
    return F.cross_entropy(logits_v, labels_v, weight=weight)

def compute_binary_loss(logits, labels, valid_mask):
    """计算二值分支损失，仅使用有监督样本。"""
    if not valid_mask.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    logits_v = logits[valid_mask]
    labels_v = labels[valid_mask]
    if logits_v.shape[2] != labels_v.shape[1] or logits_v.shape[3] != labels_v.shape[2]:
        labels_v = F.interpolate(
            labels_v.unsqueeze(1).float(),
            size=(logits_v.shape[2], logits_v.shape[3]),
            mode='nearest'
        ).squeeze(1).long()
    weight = torch.tensor([1.0, 1.0], device=logits.device)
    return F.cross_entropy(logits_v, labels_v, weight=weight)

def train(epo, model, train_loader, optimizer, scaler):
    model.train()
    for it, (images, labels, names) in enumerate(train_loader):
        images = images.cuda(args.gpu)
        labels = labels.cuda(args.gpu)
        with autocast():
            start_t = time.time()
            optimizer.zero_grad()
            hight_output, out, bin_output, edge_output = model(images)
            loss_1 = criterion(hight_output, labels)
            loss_2 = criterion(out, labels)
            boundary_labels, boundary_valid = load_boundary_labels_batch(
                names, args.boundary_dir,
                target_size=(labels.shape[1], labels.shape[2]),
                device=labels.device
            )
            loss_edge = compute_edge_loss(edge_output, boundary_labels, boundary_valid)

            binary_labels, binary_valid = load_binary_labels_batch(
                names, args.binary_dir,
                target_size=(labels.shape[1], labels.shape[2]),
                device=labels.device
            )
            loss_binary = compute_binary_loss(bin_output, binary_labels, binary_valid)
            
            loss = loss_1 + loss_2 + loss_edge + loss_binary
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        lr_this_epo = 0
        for param_group in optimizer.param_groups:
            lr_this_epo = param_group['lr']
        print('Train: %s, epo %s/%s, iter %s/%s, lr %.8f, %.2f img/sec, loss %.4f (sem1 %.4f, sem2 %.4f, edge %.4f, binary %.4f), time %s'\
              % (args.model_name, epo, args.epoch_max, it + 1, len(train_loader), lr_this_epo,
                 len(names) / (time.time() - start_t), float(loss),
                 float(loss_1), float(loss_2), float(loss_edge), float(loss_binary),
                 datetime.datetime.now().replace(microsecond=0) - start_datetime))
        if accIter['train'] % 1 == 0:
            swanlab.log({
                'Train/loss': float(loss),
                'Train/loss_semantic_1': float(loss_1),
                'Train/loss_semantic_2': float(loss_2),
                'Train/loss_edge': float(loss_edge),
                'Train/loss_binary': float(loss_binary),
                'Train/learning_rate': lr_this_epo
            }, step=accIter['train'])
        view_figure = True
        if accIter['train'] % 500 == 0:
            if view_figure:
                input_rgb_images = vutils.make_grid(images[:, :3], nrow=8, padding=10, normalize=True).float()
                swanlab.log({'Train/input_rgb_images': swanlab.Image(input_rgb_images)}, step=accIter['train'])
                
                scale = max(1, 255 // args.n_class)
                groundtruth_tensor = labels.unsqueeze(1).float() * scale
                groundtruth_tensor = torch.cat((groundtruth_tensor, groundtruth_tensor, groundtruth_tensor), 1)
                groudtruth_images = vutils.make_grid(groundtruth_tensor, nrow=8, padding=10, normalize=True)
                groudtruth_images = groudtruth_images.float()
                swanlab.log({'Train/groudtruth_images': swanlab.Image(groudtruth_images)}, step=accIter['train'])
                
                predicted_tensor = out.argmax(1).unsqueeze(1).float() * scale
                predicted_tensor = torch.cat((predicted_tensor, predicted_tensor, predicted_tensor), 1)
                predicted_images = vutils.make_grid(predicted_tensor, nrow=8, padding=10, normalize=True)
                predicted_images = predicted_images.float()
                swanlab.log({'Train/predicted_images': swanlab.Image(predicted_images)}, step=accIter['train'])
        accIter['train'] = accIter['train'] + 1

def validation(epo, model, val_loader): 
    model.eval()
    total_loss = 0.0
    num_batches = 0
    conf_total = np.zeros((args.n_class, args.n_class))
    boundary_class_ids = (
        list(range(1, args.n_class))
        if args.boundary_ignore_unlabeled else list(range(args.n_class))
    )
    boundary_stats = create_boundary_stats(args.n_class)
    with torch.no_grad():
        for it, (images, labels, names) in enumerate(val_loader):
            images = images.cuda(args.gpu)
            labels = labels.cuda(args.gpu)
            start_t = time.time()
            hight_output, out, bin_output, edge_output = model(images)
            loss_1 = criterion(hight_output, labels)
            loss_2 = criterion(out, labels)
            boundary_labels, boundary_valid = load_boundary_labels_batch(
                names, args.boundary_dir,
                target_size=(labels.shape[1], labels.shape[2]),
                device=labels.device
            )
            loss_edge = compute_edge_loss(edge_output, boundary_labels, boundary_valid)
            binary_labels, binary_valid = load_binary_labels_batch(
                names, args.binary_dir,
                target_size=(labels.shape[1], labels.shape[2]),
                device=labels.device
            )
            loss_binary = compute_binary_loss(bin_output, binary_labels, binary_valid)
            
            loss = loss_1 + loss_2 + loss_edge + loss_binary
            total_loss += float(loss)
            num_batches += 1
            
            label_map = labels.cpu().numpy()
            prediction_map = (out + hight_output).argmax(1).cpu().numpy()
            label = label_map.reshape(-1)
            prediction = prediction_map.reshape(-1)
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=list(range(args.n_class)))
            conf_total += conf
            for batch_index in range(label_map.shape[0]):
                update_boundary_stats(
                    label_map[batch_index], prediction_map[batch_index],
                    boundary_stats, args.boundary_width,
                    args.boundary_tolerance, boundary_class_ids
                )
            
            print('Val: %s, epo %s/%s, iter %s/%s, %.2f img/sec, loss %.4f, time %s' \
                  % (args.model_name, epo, args.epoch_max, it + 1, len(val_loader), len(names)/(time.time()-start_t), float(loss),
                    datetime.datetime.now().replace(microsecond=0)-start_datetime))
            if accIter['val'] % 1 == 0:
                swanlab.log({'Validation/loss': float(loss)}, step=accIter['val'])
            view_figure = False
            if accIter['val'] % 100 == 0:
                if view_figure:
                    input_rgb_images = vutils.make_grid(images[:, :3], nrow=8, padding=10, normalize=True).float()
                    swanlab.log({'Validation/input_rgb_images': swanlab.Image(input_rgb_images)}, step=accIter['val'])
                    
                    scale = max(1, 255 // args.n_class)
                    groundtruth_tensor = labels.unsqueeze(1).float() * scale
                    groundtruth_tensor = torch.cat((groundtruth_tensor, groundtruth_tensor, groundtruth_tensor), 1)
                    groudtruth_images = vutils.make_grid(groundtruth_tensor, nrow=8, padding=10, normalize=True)
                    groudtruth_images = groudtruth_images.float()
                    swanlab.log({'Validation/groudtruth_images': swanlab.Image(groudtruth_images)}, step=accIter['val'])
                    
                    predicted_tensor = out.argmax(1).unsqueeze(1).float() * scale
                    predicted_tensor = torch.cat((predicted_tensor, predicted_tensor, predicted_tensor), 1)
                    predicted_images = vutils.make_grid(predicted_tensor, nrow=8, padding=10, normalize=True)
                    predicted_images = predicted_images.float()
                    swanlab.log({'Validation/predicted_images': swanlab.Image(predicted_images)}, step=accIter['val'])
            accIter['val'] += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    
    precision, recall, IoU = compute_results(conf_total)
    mean_iou = np.mean(np.nan_to_num(IoU))
    mean_precision = np.mean(np.nan_to_num(precision))
    mean_recall = np.mean(np.nan_to_num(recall))
    boundary_iou, boundary_fscore, boundary_miou, boundary_fscore_mean = \
        calculate_boundary_metrics(boundary_stats, boundary_class_ids)
    
    validation_metrics = {
        'Validation/avg_loss': avg_loss,
        'Validation/mean_IoU': mean_iou,
        'Validation/average_precision': mean_precision,
        'Validation/average_recall': mean_recall,
        'Validation/boundary_mIoU': boundary_miou,
        'Validation/boundary_F_score': boundary_fscore_mean,
    }
    for class_id in boundary_class_ids:
        validation_metrics['Validation/boundary_IoU_class_%d' % class_id] = boundary_iou[class_id]
        validation_metrics['Validation/boundary_F_score_class_%d' % class_id] = boundary_fscore[class_id]
    swanlab.log(validation_metrics, step=epo)
    
    print('Validation Summary: loss=%.4f, mIoU=%.4f, precision=%.4f, recall=%.4f, '
          'boundary mIoU=%.4f, boundary F-score=%.4f' %
          (avg_loss, mean_iou, mean_precision, mean_recall,
           boundary_miou, boundary_fscore_mean))
    
    return avg_loss, mean_iou

def testing(epo, model, test_loader):
    model.eval()
    conf_total = np.zeros((args.n_class, args.n_class))
    boundary_class_ids = (
        list(range(1, args.n_class))
        if args.boundary_ignore_unlabeled else list(range(args.n_class))
    )
    boundary_stats = create_boundary_stats(args.n_class)
    label_list = ["unlabeled", "car", "person", "bike", "curve", "car_stop", "guardrail", "color_cone", "bump"]
    testing_results_file = os.path.join(weight_dir, 'testing_results_file.txt')
    with torch.no_grad():
        for it, (images, labels, names) in enumerate(test_loader):
            images = images.cuda(args.gpu)
            labels = labels.cuda(args.gpu)
            hight_output, out, _, edge_output = model(images)
            label_map = labels.cpu().numpy()
            prediction_map = (out + hight_output).argmax(1).cpu().numpy()
            label = label_map.reshape(-1)
            prediction = prediction_map.reshape(-1)
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=list(range(args.n_class)))
            conf_total += conf
            for batch_index in range(label_map.shape[0]):
                update_boundary_stats(
                    label_map[batch_index], prediction_map[batch_index],
                    boundary_stats, args.boundary_width,
                    args.boundary_tolerance, boundary_class_ids
                )
            print('Test: %s, epo %s/%s, iter %s/%s, time %s' % (args.model_name, epo, args.epoch_max, it+1, len(test_loader),
                 datetime.datetime.now().replace(microsecond=0)-start_datetime))
    precision, recall, IoU = compute_results(conf_total)
    boundary_iou, boundary_fscore, boundary_miou, boundary_fscore_mean = \
        calculate_boundary_metrics(boundary_stats, boundary_class_ids)
    test_metrics = {
        'Test/average_precision': precision.mean(),
        'Test/average_recall': recall.mean(),
        'Test/average_IoU': IoU.mean(),
        'Test/boundary_mIoU': boundary_miou,
        'Test/boundary_F_score': boundary_fscore_mean,
    }
    for i in range(len(precision)):
        class_name = label_list[i] if i < len(label_list) else f'class_{i}'
        test_metrics["Test(class)/precision_class_%s" % class_name] = precision[i]
        test_metrics["Test(class)/recall_class_%s" % class_name] = recall[i]
        test_metrics['Test(class)/Iou_%s' % class_name] = IoU[i]
    for class_id in boundary_class_ids:
        test_metrics['Test(class)/boundary_IoU_class_%d' % class_id] = boundary_iou[class_id]
        test_metrics['Test(class)/boundary_F_score_class_%d' % class_id] = boundary_fscore[class_id]
    swanlab.log(test_metrics, step=epo)
    if epo==0:
        with open(testing_results_file, 'w') as f:
            f.write("# %s, initial lr: %s, batch size: %s, date: %s \n" %(args.model_name, args.lr_start, args.batch_size, datetime.date.today()))
            f.write("# epoch: unlabeled, car, person, bike, curve, car_stop, guardrail, color_cone, bump, average(nan_to_num). (Acc %, IoU %)\n")
    with open(testing_results_file, 'a') as f:
        f.write(str(epo)+': ')
        for i in range(len(precision)):
            f.write('%0.4f, %0.4f, ' % (100*recall[i], 100*IoU[i]))
        f.write('%0.4f, %0.4f\n' % (100*np.mean(np.nan_to_num(recall)), 100*np.mean(np.nan_to_num(IoU))))
    print('saving testing results.')
    with open(testing_results_file, "r") as file:
        swanlab.log({'Test/testing_results': swanlab.Text(file.read())}, step=epo)

if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for training; set up a CUDA-enabled PyTorch environment.')
    torch.cuda.set_device(args.gpu)
    print("\nthe pytorch version:", torch.__version__)
    print("the gpu count:", torch.cuda.device_count())
    print("the current used gpu:", torch.cuda.current_device(), '\n')

    model = BSIFNet(n_class=args.n_class).cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_start, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay, last_epoch=-1)
    scaler = GradScaler()

    resumed_epoch = 0
    resumed_accIter = None
    resumed_best_val_miou = None
    resumed_best_val_loss = None
    resumed_best_epoch = None

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location='cuda:%d' % args.gpu, weights_only=False)
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'], strict=False)
            if 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
                for pg in optimizer.param_groups:
                    pg['lr'] = args.lr_start
                print('  Restored optimizer state, lr overridden to %.6f.' % args.lr_start)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay, last_epoch=-1)
            print('  Rebuilt scheduler with lr=%.6f, gamma=%.4f.' % (args.lr_start, args.lr_decay))
            if 'scaler' in ckpt:
                scaler.load_state_dict(ckpt['scaler'])
                print('  Restored GradScaler state.')
            if 'epoch' in ckpt:
                resumed_epoch = ckpt['epoch'] + 1
                print('  Will resume from epoch %d.' % resumed_epoch)
            if 'accIter' in ckpt:
                resumed_accIter = ckpt['accIter']
                print('  Restored accIter: %s' % resumed_accIter)
            if 'best_val_miou' in ckpt:
                resumed_best_val_miou = ckpt['best_val_miou']
                resumed_best_val_loss = ckpt.get('best_val_loss', float('inf'))
                resumed_best_epoch = ckpt.get('best_epoch', -1)
                print('  Restored best mIoU=%.4f (epoch %d).' % (resumed_best_val_miou, resumed_best_epoch))
            print('Loaded full checkpoint from: %s' % args.resume)
        else:
            state = ckpt
            model.load_state_dict(state, strict=False)
            print('Loaded model weights only from: %s (no optimizer/scheduler state)' % args.resume)

    weight_dir = os.path.join("./runs", args.model_name)
    os.makedirs(weight_dir, exist_ok=True)

    swanlab.init(
        experiment_name=args.model_name,
        description=f"Training {args.model_name} with batch_size={args.batch_size}, lr={args.lr_start}",
        config={
            'model_name': args.model_name,
            'batch_size': args.batch_size,
            'lr_start': args.lr_start,
            'lr_decay': args.lr_decay,
            'epoch_max': args.epoch_max,
            'n_class': args.n_class,
            'gpu': args.gpu,
            'boundary_width': args.boundary_width,
            'boundary_tolerance': args.boundary_tolerance,
            'boundary_ignore_unlabeled': args.boundary_ignore_unlabeled,
        }
    )
    _swanlab_log_bsifnet_code_once(step=0)

    print('training %s on GPU #%d with pytorch' % (args.model_name, args.gpu))
    print('from epoch %d / %s' % (args.epoch_from, args.epoch_max))
    print('weight will be saved in: %s' % weight_dir)

    train_dataset = MF_dataset(data_dir=args.data_dir, split='train', transform=augmentation_methods)
    val_dataset  = MF_dataset(data_dir=args.data_dir, split='val')
    test_dataset = MF_dataset(data_dir=args.data_dir, split='test')

    train_loader  = DataLoader(
        dataset     = train_dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = False
    )
    val_loader  = DataLoader(
        dataset     = val_dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = False
    )
    test_loader = DataLoader(
        dataset      = test_dataset,
        batch_size   = args.batch_size,
        shuffle      = False,
        num_workers  = args.num_workers,
        pin_memory   = True,
        drop_last    = False
    )
    start_datetime = datetime.datetime.now().replace(microsecond=0)
    accIter = resumed_accIter if resumed_accIter is not None else {'train': 0, 'val': 0}
    best_val_miou = resumed_best_val_miou if resumed_best_val_miou is not None else -1.0
    best_val_loss = resumed_best_val_loss if resumed_best_val_loss is not None else float('inf')
    best_epoch = resumed_best_epoch if resumed_best_epoch is not None else -1
    epoch_start = resumed_epoch if resumed_epoch > 0 else args.epoch_from

    for epo in range(epoch_start, args.epoch_max):
        print('\ntrain %s, epo #%s begin...' % (args.model_name, epo))
        train(epo, model, train_loader, optimizer, scaler)
        val_loss, val_miou = validation(epo, model, val_loader)

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            best_val_loss = val_loss
            best_epoch = epo
            best_model_file = os.path.join(weight_dir, 'best.pth')
            print('saving best model (val_mIoU=%.4f, val_loss=%.4f) at epoch %d to %s' % 
                  (best_val_miou, best_val_loss, best_epoch, best_model_file))
            torch.save({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'epoch': epo,
                'accIter': accIter,
                'best_val_miou': best_val_miou,
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
            }, best_model_file)
            best_info_file = os.path.join(weight_dir, 'best_info.txt')
            with open(best_info_file, 'w') as f:
                f.write('Best epoch: %d\n' % best_epoch)
                f.write('Best validation mIoU: %.4f\n' % best_val_miou)
                f.write('Best validation loss: %.4f\n' % best_val_loss)
            swanlab.log({
                'Best/best_val_mIoU': best_val_miou,
                'Best/best_val_loss': best_val_loss,
                'Best/best_epoch': best_epoch
            }, step=epo)
        else:
            print('Current val_mIoU=%.4f, best val_mIoU=%.4f at epoch %d (not saving best)' % 
                  (val_miou, best_val_miou, best_epoch))
            swanlab.log({
                'Best/best_val_mIoU': best_val_miou,
                'Best/best_val_loss': best_val_loss,
                'Best/best_epoch': best_epoch
            }, step=epo)

        scheduler.step()

        latest_file = os.path.join(weight_dir, 'latest.pth')
        torch.save({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'epoch': epo,
            'accIter': accIter,
            'best_val_miou': best_val_miou,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
        }, latest_file)
