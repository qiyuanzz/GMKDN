import numpy as np
import torch
import os




def train_loop_classification_coattn(epoch, model, loader, optimizer, scheduler, AUROC, AP, metrics, writer=None, loss_fn=None,  gc=16, args=None):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    train_loss_surv, train_loss = 0., 0.
    metrics = metrics.to(device)
    AP = AP.to(device)
    AUROC = AUROC.to(device)
    # Y_hats = []
    # labels = []
    # Y_probs = []
    print('\n')
    
    for batch_idx, (data_WSI, label) in enumerate(loader):

        
        data_WSI = data_WSI.to(device)
        label = label.type(torch.LongTensor).to(device)
        logits, Y_prob, Y_hat, total_inst_loss = model(x_path=data_WSI, label=label)
        loss_class = loss_fn(logits, label)

        AUROC.update(Y_prob[:,1], label.squeeze())
        metrics.update(Y_hat, label)
        AP.update(Y_prob[:,1], label)
        loss = loss_class * 0.8 + total_inst_loss * 0.2
        loss_value = loss.item()

        train_loss += loss_value

        # loss = loss / gc + loss_reg
        loss = loss / gc
        loss.backward()
        # loss.backward(retain=True)

        if (batch_idx + 1) % gc == 0: 
            optimizer.step()
            optimizer.zero_grad()


    train_loss /= len(loader)
    auroc = AUROC.compute()
    metrics = metrics.compute()
    ap = AP.compute()
    scheduler.step()
    # class_accuracies = {}
    # for cls in range(args.n_classes):
    #     cls_indices = labels == cls
    #     class_accuracies[cls] = accuracy_score(labels[cls_indices], Y_hats[cls_indices])
    train_epoch_str = 'Epoch: {}, train_loss: {:.4f}, auc: {:.4f}, ap: {:.4f}, BinaryAccuracy: {:.4f}, BinaryPrecision: {:.4f}, BinaryRecall: {:.4f} BinarySpecificity: {:.4f}, BinaryF1Score: {:.4f}'\
        .format(epoch, train_loss, auroc, ap, metrics['BinaryAccuracy'], metrics['BinaryPrecision'], metrics['BinaryRecall'], metrics['BinarySpecificity'], metrics['BinaryF1Score'])
    print(train_epoch_str)
    with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
        f.write(train_epoch_str+'\n')
    f.close()

    if writer:
        # writer.add_scalar('train/loss_surv', train_loss_surv, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)


def validate_classification_coattn(cur, epoch, model, loader, AUROC, AP, metrics, early_stopping=None, writer=None, loss_fn=None, results_dir=None, args=None):
    model.eval()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    val_loss = 0.
    metrics = metrics.to(device)
    AUROC = AUROC.to(device)
    AP = AP.to(device)


    for batch_idx, (data_WSI, label) in enumerate(loader):

        data_WSI = data_WSI.cuda()
        label = label.type(torch.LongTensor).cuda()


        with torch.no_grad():
            logits, Y_prob, Y_hat, total_inst_loss = model(x_path=data_WSI, label=label) # return hazards, S, Y_hat, A_raw, results_dict

        loss_class = loss_fn(logits, label)
        AUROC.update(Y_prob[:,1], label.squeeze())
        metrics.update(Y_hat, label)
        AP.update(Y_prob[:,1], label)
        loss = loss_class * 0.8 + total_inst_loss * 0.2
        loss_value = loss.item()
        val_loss += loss_value



    val_loss /= len(loader)
    auroc = AUROC.compute()
    metrics = metrics.compute()
    ap = AP.compute()
    val_epoch_str = 'Epoch: {}, val_loss: {:.4f}, auc: {:.4f}, ap: {:.4f}, BinaryAccuracy: {:.4f}, BinaryPrecision: {:.4f}, BinaryRecall: {:.4f}, BinarySpecificity: {:.4f}, BinaryF1Score: {:.4f}'\
        .format(epoch, val_loss, auroc, ap, metrics['BinaryAccuracy'], metrics['BinaryPrecision'], metrics['BinaryRecall'], metrics['BinarySpecificity'], metrics['BinaryF1Score'])
    print(val_epoch_str)
    with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
        f.write(val_epoch_str+'\n')
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)

    if early_stopping:
        assert results_dir
        if args.train_mode == 'auc':
            early_stopping(epoch, auroc, model, ckpt_name=os.path.join(args.results_dir, "s_{}_max_auc_checkpoint.pt".format(cur)))
        elif args.train_mode == 'loss':
            early_stopping(epoch, val_loss, model, ckpt_name=os.path.join(args.results_dir, "s_{}_min_loss_checkpoint.pt".format(cur)))
        else:
            raise ValueError("train_mode should be 'auc' or 'loss'")
        
        if early_stopping.early_stop:
            print("Early stopping")
            best_model = model
            return val_loss, val_loss, ap, metrics, True

    return val_loss, auroc, ap, metrics, False


def test_classification_coattn(model, loader, AUROC, AP, metrics, writer=None, loss_fn=None, args=None):
    model.eval()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    val_loss = 0.
    metrics = metrics.to(device)
    AUROC = AUROC.to(device)
    AP = AP.to(device)

    # slide_ids = loader.dataset.slide_data['slide_id']

    for batch_idx, (data_WSI, label) in enumerate(loader):

        data_WSI = data_WSI.cuda()
        label = label.type(torch.LongTensor).cuda()

        with torch.no_grad():
            logits, Y_prob, Y_hat, total_inst_loss = model(x_path=data_WSI, label=label) # return hazards, S, Y_hat, A_raw, results_dict

        loss_class = loss_fn(logits, label)
        AUROC.update(Y_prob[:,1], label.squeeze())
        metrics.update(Y_hat, label)
        AP.update(Y_prob[:,1], label)
        loss = loss_class * 0.8 + total_inst_loss * 0.2
        loss_value = loss.item()
        val_loss += loss_value


    val_loss /= len(loader)
    auroc = AUROC.compute()
    metrics = metrics.compute()
    ap = AP.compute()
    val_epoch_str = 'test_loss: {:.4f}, auc: {:.4f}, ap: {:.4f}, BinaryAccuracy: {:.4f}, BinaryPrecision: {:.4f}, BinaryRecall: {:.4f}, BinarySpecificity: {:.4f}, BinaryF1Score: {:.4f}, BinaryCohensKappa: {:.4f}'\
        .format( val_loss, auroc, ap, metrics['BinaryAccuracy'], metrics['BinaryPrecision'], \
                metrics['BinaryRecall'], metrics['BinarySpecificity'], metrics['BinaryF1Score'], metrics['BinaryCohenKappa'])
    print(val_epoch_str)
    with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
        f.write(val_epoch_str+'\n')
    if writer:
        # writer.add_scalar('val/loss_surv', val_loss_surv, epoch)
        writer.add_scalar('test/loss', val_loss)
        # writer.add_scalar('val/c-index', c_index, epoch)


    return val_loss, auroc, ap, metrics, False