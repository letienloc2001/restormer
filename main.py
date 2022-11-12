import os

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Restormer
from utils import parse_args, RainDataset, rgb_to_y, psnr, ssim


def test_loop(net, data_loader, num_iter, model_file='', result_path="result", data_name='rain100L'):
    net.eval()
    total_psnr, total_ssim, count = 0.0, 0.0, 0
    with torch.no_grad():
        test_bar = tqdm(data_loader, initial=1, dynamic_ncols=True)
        for rain, norain, name, h, w in test_bar:
            rain, norain = rain.cuda(), norain.cuda()
            out = torch.clamp((torch.clamp(net(rain)[:, :, :h, :w], 0, 1).mul(255)), 0, 255).byte()
            norain = torch.clamp(norain[:, :, :h, :w].mul(255), 0, 255).byte()
            # computer the metrics with Y channel and double precision
            y, gt = rgb_to_y(out.double()), rgb_to_y(norain.double())
            current_psnr, current_ssim = psnr(y, gt), ssim(y, gt)
            total_psnr += current_psnr.item()
            total_ssim += current_ssim.item()
            count += 1
            save_path = '{}/{}/{}'.format(result_path, data_name, name[0])
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            Image.fromarray(out.squeeze(dim=0).permute(1, 2, 0).contiguous().cpu().numpy()).save(save_path)
            test_bar.set_description('Test Iter: [{}/{}] PSNR: {:.2f} SSIM: {:.3f}'
                                     .format(num_iter, 1 if model_file else num_iter,
                                             total_psnr / count, total_ssim / count))
    return total_psnr / count, total_ssim / count


def save_loop(net, data_loader, num_iter, results, best_psnr, best_ssim, model_file='', data_name='rain100L', result_path='result'):
    val_psnr, val_ssim = test_loop(net, data_loader, num_iter, model_file=model_file, data_name=data_name, result_path=result_path)
    results['PSNR'].append('{:.2f}'.format(val_psnr))
    results['SSIM'].append('{:.3f}'.format(val_ssim))
    # save statistics
    data_frame = pd.DataFrame(data=results, index=range(1, (num_iter if model_file else num_iter // 1000) + 1))
    data_frame.to_csv('{}/{}.csv'.format(result_path, data_name), index_label='Iter', float_format='%.3f')
    if val_psnr > best_psnr and val_ssim > best_ssim:
        best_psnr, best_ssim = val_psnr, val_ssim
        with open('{}/{}.txt'.format(result_path, data_name), 'w') as f:
            f.write('Iter: {} PSNR:{:.2f} SSIM:{:.3f}'.format(num_iter, best_psnr, best_ssim))
        torch.save(net.state_dict(), '{}/{}.pth'.format(result_path, data_name))
    print(best_psnr, best_ssim)
    return best_psnr, best_ssim


# if __name__ == '__main__':
#     args = parse_args()
#     test_dataset = RainDataset(args.data_path, args.data_name, 'test')
#     test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.workers)

#     results, best_psnr, best_ssim = {'PSNR': [], 'SSIM': []}, 0.0, 0.0
#     model = Restormer(args.num_blocks, args.num_heads, args.channels, args.num_refinement, args.expansion_factor).cuda()
#     if args.model_file:
#         model.load_state_dict(torch.load(args.model_file))
#         save_loop(model, test_loader, 1)
#     else:
#         optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
#         lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.num_iter, eta_min=1e-6)
#         total_loss, total_num, results['Loss'], i = 0.0, 0, [], 0
#         train_bar = tqdm(range(1, args.num_iter + 1), initial=1, dynamic_ncols=True)
#         for n_iter in train_bar:
#             # progressive learning
#             if n_iter == 1 or n_iter - 1 in args.milestone:
#                 end_iter = args.milestone[i] if i < len(args.milestone) else args.num_iter
#                 start_iter = args.milestone[i - 1] if i > 0 else 0
#                 length = args.batch_size[i] * (end_iter - start_iter)
#                 train_dataset = RainDataset(args.data_path, args.data_name, 'train', args.patch_size[i], length)
#                 train_loader = iter(DataLoader(train_dataset, args.batch_size[i], True, num_workers=args.workers))
#                 i += 1
#             # train
#             model.train()
#             rain, norain, name, h, w = next(train_loader)
#             rain, norain = rain.cuda(), norain.cuda()
#             out = model(rain)
#             loss = F.l1_loss(out, norain)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             total_num += rain.size(0)
#             total_loss += loss.item() * rain.size(0)
#             train_bar.set_description('Train Iter: [{}/{}] Loss: {:.3f}'
#                                       .format(n_iter, args.num_iter, total_loss / total_num))

#             lr_scheduler.step()
#             if n_iter % 100 == 0: #1000
#                 results['Loss'].append('{:.3f}'.format(total_loss / total_num))
#                 save_loop(model, test_loader, n_iter)


def train(
        data_path, 
        data_name='rain100L',
        lr=0.0003,
        num_iter=300000,
        batch_size=(64, 40, 32, 16, 8, 8),
        patch_size=(128, 160, 192, 256, 320, 384),
        milestone=(92000, 156000, 204000, 240000, 276000),
        weight_decay=1e-4,
        model_file="",
        num_blocks=(4, 6, 6, 8),
        num_heads=(1, 2, 4, 8),
        channels=(48, 96, 192, 384), 
        num_refinement=4, 
        expansion_factor=2.66,
        workers=8):
    test_dataset = RainDataset(data_path, data_name, 'test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=workers)

    results, best_psnr, best_ssim = {'PSNR': [], 'SSIM': []}, 0.0, 0.0
    model = Restormer(num_blocks, num_heads, channels, num_refinement, expansion_factor).cuda()
    if model_file:
        model.load_state_dict(torch.load(model_file))
        best_psnr, best_ssim = save_loop(model, test_loader, 1, results, best_psnr, best_ssim)
    else:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_iter, eta_min=1e-6)
        total_loss, total_num, results['Loss'], i = 0.0, 0, [], 0
        train_bar = tqdm(range(1, num_iter + 1), initial=1, dynamic_ncols=True)
        for n_iter in train_bar:
            # progressive learning
            if n_iter == 1 or n_iter - 1 in milestone:
                end_iter = milestone[i] if i < len(milestone) else num_iter
                start_iter = milestone[i - 1] if i > 0 else 0
                length = batch_size[i] * (end_iter - start_iter)
                train_dataset = RainDataset(data_path, data_name, 'train', patch_size[i], length)
                train_loader = iter(DataLoader(train_dataset, batch_size[i], True, num_workers=workers))
                i += 1
            # train
            model.train()
            rain, norain, name, h, w = next(train_loader)
            rain, norain = rain.cuda(), norain.cuda()
            out = model(rain)
            loss = F.l1_loss(out, norain)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_num += rain.size(0)
            total_loss += loss.item() * rain.size(0)
            train_bar.set_description('Train Iter: [{}/{}] Loss: {:.3f}'
                                      .format(n_iter, num_iter, total_loss / total_num))

            lr_scheduler.step()
            if n_iter % 1000 == 0: #1000
                results['Loss'].append('{:.3f}'.format(total_loss / total_num))
                best_psnr, best_ssim = save_loop(model, test_loader, n_iter, results, best_psnr, best_ssim)

