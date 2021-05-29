from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('../log')

for epoch in range(100):
    mAP = epoch*2
    writer.add_scalar('mAP', mAP, epoch)