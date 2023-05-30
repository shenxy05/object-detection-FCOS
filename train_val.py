from model.fcos import FCOSDetector
import torch
from dataset.VOC_dataset import VOCDataset
import math, time
from dataset.augment import Transforms
import os
import cv2
import numpy as np
import random
import torch.backends.cudnn as cudnn
import argparse
from eval_voc import sort_by_score, eval_ap_2d
from torch.utils.tensorboard import SummaryWriter
from detect import realtime_vis


def test_model(eval_loader,model):
    gt_boxes=[]
    gt_classes=[]
    pred_boxes=[]
    pred_classes=[]
    pred_scores=[]
    num=0
    for img,boxes,classes in eval_loader:
        with torch.no_grad():
            out=model(img.cuda())
            pred_boxes.append(out[2][0].cpu().numpy())
            pred_classes.append(out[1][0].cpu().numpy())
            pred_scores.append(out[0][0].cpu().numpy())
        gt_boxes.append(boxes[0].numpy())
        gt_classes.append(classes[0].numpy())
        num+=1
        print(num,end='\r')

    pred_boxes,pred_classes,pred_scores=sort_by_score(pred_boxes,pred_classes,pred_scores)
    all_AP=eval_ap_2d(gt_boxes,gt_classes,pred_boxes,pred_classes,pred_scores,0.5,len(eval_dataset.CLASSES_NAME))
    print("all classes AP=====>\n")
    for key,value in all_AP.items():
        print('ap for {} is {}'.format(eval_dataset.id2name[int(key)],value))
    mAP=0.
    for class_id,class_mAP in all_AP.items():
        mAP+=float(class_mAP)
    mAP/=(len(eval_dataset.CLASSES_NAME)-1)
    print("mAP=====>%.3f\n"%mAP)
    return mAP


class Config():
    #backbone
    pretrained=True
    freeze_stage_1=True
    freeze_bn=True

    #fpn
    fpn_out_channels=256
    use_p5=True

    #head
    class_num=80
    use_GN_head=True
    prior=0.01
    add_centerness=True
    cnt_on_reg=True

    #training
    strides=[8,16,32,64,128]
    limit_range=[[-1,64],[64,128],[128,256],[256,512],[512,999999]]

    #inference
    score_threshold=0.25
    nms_iou_threshold=0.6
    max_detection_boxes_num=300
    
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--n_gpu", type=str, default='0,1', help="number of cpu threads to use during batch generation")
    opt = parser.parse_args()
    writer=SummaryWriter(log_dir='runs/FCOS')
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.n_gpu
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(0)
    transform = Transforms()
    train_dataset = VOCDataset(root_dir='./VOC2012',resize_size=[800,1333],
                            split='train',use_difficult=True,is_train=True,augment=transform)

    model = FCOSDetector(mode="training").cuda()
    model = torch.nn.DataParallel(model)
    # model.load_state_dict(torch.load('/mnt/cephfs_new_wj/vc/zhangzhenghao/FCOS.Pytorch/output1/model_6.pth'))



    eval_dataset = VOCDataset(root_dir='./VOC2012', resize_size=[800, 1333],
                               split='test', use_difficult=False, is_train=False, augment=None)
    print("INFO===>eval dataset has %d imgs"%len(eval_dataset))
    eval_loader=torch.utils.data.DataLoader(eval_dataset,batch_size=1,shuffle=False,collate_fn=eval_dataset.collate_fn,num_workers=opt.n_cpu)

    model_val=FCOSDetector(mode="inference", config=Config)
    model_val = torch.nn.DataParallel(model_val)
    # model.load_state_dict(torch.load("./checkpoint/voc_78.7.pth",map_location=torch.device('cpu')))
    model_val=model_val.cuda().eval()
    print("===>success loading model")

    root="./visualize/"
    names=os.listdir(root)
    imgs=[]
    for name in names:
        imgs.append(cv2.imread(root+name))


    BATCH_SIZE = opt.batch_size
    EPOCHS = opt.epochs
    #WARMPUP_STEPS_RATIO = 0.12
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                            collate_fn=train_dataset.collate_fn,
                                            num_workers=opt.n_cpu, worker_init_fn=np.random.seed(0))
    print("total_images : {}".format(len(train_dataset)))
    steps_per_epoch = len(train_dataset) // BATCH_SIZE
    TOTAL_STEPS = steps_per_epoch * EPOCHS
    WARMPUP_STEPS = 601

    GLOBAL_STEPS = 1
    LR_INIT = 2e-3
    LR_END = 2e-5
    optimizer = torch.optim.SGD(model.parameters(),lr =LR_INIT,momentum=0.9,weight_decay=0.0001)

# def lr_func():
#      if GLOBAL_STEPS < WARMPUP_STEPS:
#          lr = GLOBAL_STEPS / WARMPUP_STEPS * LR_INIT
#      else:
#          lr = LR_END + 0.5 * (LR_INIT - LR_END) * (
#              (1 + math.cos((GLOBAL_STEPS - WARMPUP_STEPS) / (TOTAL_STEPS - WARMPUP_STEPS) * math.pi))
#          )
#      return float(lr)


    model.train()

    tb_iter=0
    for epoch in range(EPOCHS):
        for epoch_step, data in enumerate(train_loader):
            batch_imgs, batch_boxes, batch_classes = data
            batch_imgs = batch_imgs.cuda()
            batch_boxes = batch_boxes.cuda()
            batch_classes = batch_classes.cuda()

        #lr = lr_func()
            if GLOBAL_STEPS < WARMPUP_STEPS:
                lr = float(GLOBAL_STEPS / WARMPUP_STEPS * LR_INIT)
                for param in optimizer.param_groups:
                    param['lr'] = lr
            if GLOBAL_STEPS == 24001:
                lr = LR_INIT * 0.1
                for param in optimizer.param_groups:
                    param['lr'] = lr
            if GLOBAL_STEPS == 32401:
                lr = LR_INIT * 0.01
                for param in optimizer.param_groups:
                    param['lr'] = lr
            start_time = time.time()

            optimizer.zero_grad()
            losses = model([batch_imgs, batch_boxes, batch_classes])
            loss = losses[-1]
            loss.mean().backward()
            optimizer.step()

            end_time = time.time()
            cost_time = int((end_time - start_time) * 1000)
            if epoch_step % 20 == 0:
                print("global_steps:%d epoch:%d steps:%d/%d cls_loss:%.4f cnt_loss:%.4f reg_loss:%.4f cost_time:%dms lr=%.4e total_loss:%.4f" % \
                (GLOBAL_STEPS, epoch + 1, epoch_step + 1, steps_per_epoch, losses[0].mean(), losses[1].mean(),
                 losses[2].mean(), cost_time, lr, loss.mean()))


            GLOBAL_STEPS += 1
            if GLOBAL_STEPS % 60 == 0:
                step = GLOBAL_STEPS // 60
                writer.add_scalar("cls_loss",losses[0].mean(),step)
                writer.add_scalar("cnt_loss",losses[1].mean(),step)
                writer.add_scalar("reg_loss",losses[2].mean(),step)
                writer.add_scalar("total_loss",loss.mean(),step)
                writer.add_scalar("learning rate",lr,step)

            if (epoch_step + 1) % 170 == 0:
                tb_iter += 1
                torch.save(model.state_dict(),
                        "./checkpoint/model_{}.pth".format(epoch + 1))
                model_val.load_state_dict(torch.load("./checkpoint/model_{}.pth".format(epoch + 1)))
                with torch.no_grad():
                    map = test_model(eval_loader, model_val)
                    writer.add_scalar("mAP", map,tb_iter)
                    test_loss = 0
                    for i, data in enumerate(eval_loader):

                        batch_imgs, batch_boxes, batch_classes = data
                        batch_imgs = batch_imgs.cuda()
                        batch_boxes = batch_boxes.cuda()
                        batch_classes = batch_classes.cuda()
                        
                        loss = model([batch_imgs, batch_boxes, batch_classes])[-1]
                        test_loss += loss
                    test_loss /= len(eval_dataset)
                    writer.add_scalar("test loss", test_loss.item(), tb_iter)
                    print("-------------------test loss=", test_loss.item())
                    fig = realtime_vis(model_val, imgs)
                    if fig is not None:
                        writer.add_figure("detection test", figure=fig, global_step=tb_iter)
                    torch.cuda.empty_cache()












