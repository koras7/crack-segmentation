#General libraries
import time
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import scipy
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from .losses import WeightedBCEDiceLoss

import torchcubicspline

from torchcubicspline import(natural_cubic_spline_coeffs, 
                             NaturalCubicSpline)

#Custom libraries
from utils import Select_times_function, EarlyStopping, SaveBestModel, to_np, Integral_part, LRScheduler, load_checkpoint, Train_val_split, Dynamics_Dataset, Test_Dynamics_Dataset
from torch.utils.data import SubsetRandomSampler
from IE_source.solver import IESolver_monoidal
from IE_source.Attentional_IE_solver import Integral_spatial_attention_solver_multbatch
from IE_source.kernels import RunningAverageMeter, log_normal_pdf, normal_kl
from utils import plot_reconstruction

#Torch libraries
import torch
from torch.nn import functional as F

if torch.cuda.is_available():  
    device = "cuda:0" 
else:  
    device = "cpu"

                
                
def Crack_experiment(model, Encoder, Decoder, dataloaders, time_seq, index_np, mask, times, args, extrapolation_points): # experiment_name, plot_freq=1):
    # scaling_factor=1
    import numpy as np
    import matplotlib.pyplot as plt
    
    
    #metadata for saving checkpoints
    if args.model=='nie': 
        str_model_name = "nie"
    elif args.model=='node': 
        str_model_name = "node"
    
    str_model = f"{str_model_name}"
    str_log_dir = args.root_path
    path_to_experiment = os.path.join(str_log_dir,str_model_name, args.experiment_name)

    if args.mode=='train':
        if not os.path.exists(path_to_experiment):
            os.makedirs(path_to_experiment)

        
        print('path_to_experiment: ',path_to_experiment)
        txt = os.listdir(path_to_experiment)

        runs = []
        for i in txt:
            if i.startswith('run'):
                try:
                    _ = int(i[3:])  # check suffix
                    runs.append(i)
                except ValueError:
                    continue


        if len(runs) == 0:
            num_experiments=0
        else: 
            num_experiments = [int(i[3:]) for i in runs]
            num_experiments = np.array(num_experiments).max()
         # -- logger location
        #writer = SummaryWriter(os.path.join(path_to_experiment,'run'+str(num_experiments+1)))
        #print('writer.log_dir: ',writer.log_dir)
        
        path_to_save_plots = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'plots')
        path_to_save_models = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'model')
        if not os.path.exists(path_to_save_plots):
            os.makedirs(path_to_save_plots)
        if not os.path.exists(path_to_save_models):
            os.makedirs(path_to_save_models)
            
        #with open(os.path.join(writer.log_dir,'commandline_args.txt'), 'w') as f:
        #    for key, value in args.__dict__.items(): 
        #        f.write('%s:%s\n' % (key, value))



    dataloaders= dataloaders
    times = time_seq
    
    
    if Encoder is None and Decoder is None:
        All_parameters = model.parameters()
    elif Encoder is not None and Decoder is None:
        All_parameters = list(model.parameters())+list(Encoder.parameters())
    elif Decoder is not None and Encoder is None:
        All_parameters = list(model.parameters())+list(Decoder.parameters())
    else:
        All_parameters = list(model.parameters())+list(Encoder.parameters())+list(Decoder.parameters())
    
    optimizer = torch.optim.Adam(All_parameters, lr=args.lr, weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=0,last_epoch=-1)# Emanuele's version
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,last_epoch=-1) #My first version
    #scheduler = LRScheduler(optimizer,patience = 20,min_lr=1e-12,factor=0.1)#torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0,last_epoch=-1)
    if args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.plat_patience, min_lr=args.min_lr, factor=args.factor)
    elif args.lr_scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.min_lr,last_epoch=-1)

    # optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    if args.resume_from_checkpoint is not None:
        path = os.path.join(args.root_path,args.model,args.experiment_name,args.resume_from_checkpoint,'model')
        
        if Encoder is None or Decoder is None:
            model, optimizer, scheduler, pos_enc, pos_dec, f_func = load_checkpoint(path, model, optimizer, scheduler, None, None,  None)
        else:
            G_NN, optimizer, scheduler, model, Encoder, Decoder = load_checkpoint(path, None, optimizer, scheduler, model, Encoder, Decoder)

    
    if args.eqn_type=='Navier-Stokes':
        spatial_domain_xy = torch.meshgrid([torch.linspace(0,1,args.n_points) for i in range(2)])
        
        x_space = spatial_domain_xy[0].flatten().unsqueeze(-1)
        y_space = spatial_domain_xy[1].flatten().unsqueeze(-1)
        
        spatial_domain = torch.cat([x_space,y_space],-1)
    
    
    if args.mode=='train':
        #lr_scheduler = LRScheduler(optimizer,patience = 50,min_lr=1e-5,factor=0.1)
        early_stopping = EarlyStopping(patience=1000,min_delta=0)

        # Loss_print = []
        # Val_Loss = []
        all_train_loss=[]
        all_val_loss=[]
        
            
 
        
        # Train Neural IDE
        get_times = Select_times_function(times,extrapolation_points)

        save_best_model = SaveBestModel()
        start = time.time()

        # loss_func = torch.nn.CrossEntropyLoss()
       
        # loss_func = torch.nn.CrossEntropyLoss(weight=args.class_weights.to(args.device))
        
        # loss_func = FocalLoss(alpha=args.class_weights, gamma=2)
        
        # loss_func = DiceLoss()
        
        # alpha = torch.tensor(args.class_weights).to(args.device)
        # loss_func = FocalDiceLoss(alpha=alpha, gamma=2.0, dice_weight=1.0, focal_weight=1.0)
        
        class_weights = args.class_weights.to(args.device)
        loss_func = WeightedBCEDiceLoss(class_weights=class_weights, alpha = args.alpha)
        
                



        

        
        
    
            
        for i in range(args.epochs):
            
         
            
            model.train()
            if Encoder is not None:
                Encoder.train()
            if Decoder is not None:
                Decoder.train()
            
            start_i = time.time()
            print('Epoch:',i)
            # import GPUtil
            # GPUtil.showUtilization()
            counter=0
            train_loss = 0.0
            # print(f"Using Focal Loss with gamma=2 and alpha={args.class_weights}")

            scaler = torch.cuda.amp.GradScaler()


            
           
                
            for obs_,masks_ in tqdm(dataloaders['train']):

                # images = images.to(device, non_blocking=True)
                # masks = masks.to(device, non_blocking=True)


        

        
                
                obs_ = obs_.to(args.device,non_blocking=True)

                masks_ = masks_.to(args.device,non_blocking=True)

                # c = lambda x: \
                # Encoder(obs_[:,:,:,:1,:].requires_grad_(True))\
                #     .repeat(1,1,1,args.time_points,1).to(args.device)
            
                # y_0 = Encoder(obs_[:,:,:,:1,:])  # Already [B, H, W, 1, C]

                
                                          

                # # # This one orginal
                # c= lambda x: \
                # Encoder(obs_[:,:,:,0,:].permute(0,3,1,2).requires_grad_(True))\
                #         .permute(0,2,3,1).unsqueeze(-2)\
                #         .contiguous().repeat(1,1,1,args.time_points,1).to(args.device)

                # # y_0 = Encoder(obs_[:,:,:,:1,:])  # [B, H, W, 1, C]
 
                    
                # y_0 =  Encoder(obs_[:,:,:,0,:].permute(0,3,1,2))\
                #             .permute(0,2,3,1).unsqueeze(-2)


                enc = Encoder(obs_[:, :, :, 0, :].permute(0, 3, 1, 2))
                enc = enc.permute(0, 2, 3, 1).unsqueeze(-2).contiguous()
            
                y_0 = enc
                c = lambda x: enc.repeat(1, 1, 1, args.time_points, 1)


                
                # print(c(1).shape)  
                # GPUtil.showUtilization()
                if args.ts_integration is not None:
                    times_integration = args.ts_integration
                else:
                    times_integration = torch.linspace(0,1,args.time_points)
                    
                with torch.cuda.amp.autocast(dtype=torch.float16):
                
                    z_ = Integral_spatial_attention_solver_multbatch(
                                        times_integration.to(args.device),
                                        y_0.to(args.device),
                                        y_init= None,
                                        c=c,
                                        sampling_points = args.time_points,
                                        mask=mask,
                                        Encoder = model,
                                        max_iterations = args.max_iterations,
                                        spatial_integration=True,
                                        spatial_domain= spatial_domain.to(args.device),
                                        spatial_domain_dim=2,
                                        #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                        #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                        smoothing_factor=args.smoothing_factor,
                                        use_support=False,
                                        accumulate_grads=True,
                                        initialization=args.initialization
                                        ).solve()
                    
                
                # z_= Decoder(z_.squeeze()) #changed for new decoder


              
                     
                #loss_ts_ = get_times.select_times(ts_)[1]
                

                # z_= z_.permute(0,3,1,2) #changed for new docoder
                # masks_=masks_.squeeze(-1) changed for new decoder

                #New added 

                # Reshape for new UNet decoder: [B, H, W, C] → [B, C, H, W]
                z_decoder_input = z_.squeeze(-2).permute(0, 3, 1, 2)  # → [B, 16, 128, 128]
                z_ = Decoder(z_decoder_input)  # → [B, 2, 256, 256]
                
                # z_ already in [B, C, H, W] format - no need to permute!
                masks_ = masks_.squeeze(-1)  # [B, H, W]

                ##till here

                

                #print(masks_.shape,z_.shape)

                # n_classes = len(torch.unique(masks_))
                # print(f"Number of classes: {n_classes}")


                # masks_ = torch.clamp(masks_, min=0, max=19-1)

                # # Before loss calculation
                # print(f"Unique values in masks_: {torch.unique(masks_)}")
                #print("Unique values in masks_: ", torch.unique(masks_))
                #print("z_.shape: ", z_.shape)

                #masks_ = (masks_ > 0).long()
                



                
                loss = loss_func(z_, masks_) #Original 
                # print('z_[:,:].to(args.device): ',z_[:,:].to(args.device))
                # print('obs_.to(args.device).detach()[:,:]: ',obs_.to(args.device).detach()[:,:])
                # loss = F.mse_loss(z_[:,:].to(args.device), obs_.to(args.device).detach()[:,:]) #Original 

                
                # ###############################
                # Loss_print.append(to_np(loss))
                # ###############################

                # optimizer.zero_grad()
                # loss.backward()#(retain_graph=True)
                # optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()


                # n_iter += 1
                counter += 1
                train_loss += loss.item()
                
            if i>15 and args.lr_scheduler == 'CosineAnnealingLR':
                scheduler.step()
                
                
            train_loss /= counter
            all_train_loss.append(train_loss)
            if  args.lr_scheduler != 'CosineAnnealingLR':
                scheduler.step(train_loss)
                   
            del train_loss, loss, obs_,  z_, masks_

            ## Validating
                
            model.eval()
            if Encoder is not None:
                Encoder.eval()
            if Decoder is not None:
                Decoder.eval()
                
            with torch.no_grad():

                    #Only do this if there is a validation dataset
                
                val_loss = 0.0
                counter = 0

                
                for obs_val, masks_val in tqdm(dataloaders['val']):
                
                



        
                
                    obs_val = obs_val.to(args.device,non_blocking=True)

                    masks_val = masks_val.to(args.device,non_blocking=True)
        
        
        
        
                                              
        
                    
                    # c= lambda x: \
                    # Encoder(obs_val[:,:,:,0,:].permute(0,3,1,2).requires_grad_(True))\
                    #         .permute(0,2,3,1).unsqueeze(-2)\
                    #         .contiguous().repeat(1,1,1,args.time_points,1).to(args.device)
                        
                    
        
                        
                    # y_0 =  Encoder(obs_val[:,:,:,0,:].permute(0,3,1,2))\
                    #             .permute(0,2,3,1).unsqueeze(-2)
                    # #print(c(1).shape)    

                    enc = Encoder(obs_val[:, :, :, 0, :].permute(0, 3, 1, 2))
                    enc = enc.permute(0, 2, 3, 1).unsqueeze(-2).contiguous()
                
                    y_0 = enc
                    c = lambda x: enc.repeat(1, 1, 1, args.time_points, 1)
                    if args.ts_integration is not None:
                        times_integration = args.ts_integration
                    else:
                        times_integration = torch.linspace(0,1,args.time_points)
                    
                    z_val = Integral_spatial_attention_solver_multbatch(
                                        times_integration.to(args.device),
                                        y_0.to(args.device),
                                        y_init= None,
                                        c=c,
                                        sampling_points = args.time_points,
                                        mask=mask,
                                        Encoder = model,
                                        max_iterations = args.max_iterations,
                                        spatial_integration=True,
                                        spatial_domain= spatial_domain.to(args.device),
                                        spatial_domain_dim=2,
                                        #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                        #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                        smoothing_factor=args.smoothing_factor,
                                        use_support=False,
                                        accumulate_grads=True,
                                        initialization=args.initialization
                                        ).solve()
                    
                    
                    # z_val= Decoder(z_val.squeeze())

                    # z_val= z_val.permute(0,3,1,2)
                    # masks_val=masks_val.squeeze(-1)

                    ################

                    # Reshape for new UNet decoder
                    z_decoder_input = z_val.squeeze(-2).permute(0, 3, 1, 2)  # → [B, 16, 128, 128]
                    z_val = Decoder(z_decoder_input)  # → [B, 2, 256, 256]
                    
                    # z_val already in correct format
                    masks_val = masks_val.squeeze(-1)  # [B, H, W]

                    #################

                    #masks_val = masks_val.clone()
                    #masks_val[masks_val != 0] = 1
                         
                    #loss_ts_ = get_times.select_times(ts_)[1]
                    
                    loss_validation = loss_func(z_val, masks_val) #Original
                            # Val_Loss.append(to_np(loss_validation))
                        
                    del obs_val, z_val, masks_val
 
                    counter += 1
                    val_loss += loss_validation.item()
                    
                    del loss_validation

                    #LRScheduler(loss_validation)
                    if args.lr_scheduler == 'ReduceLROnPlateau':
                        scheduler.step(val_loss)
                
                
               

                val_loss /= counter
                all_val_loss.append(val_loss)
                
                del val_loss

                if i % args.plot_freq == 0 and i != 0:
                    
                    plt.figure(0, figsize=(8,8),facecolor='w')
                    # plt.plot(np.linspace(0,len(Loss_print),len(Loss_print)),np.log10(Loss_print))
                    # plt.plot(np.linspace(0,len(Val_Loss),len(Val_Loss)),np.log10(Val_Loss))
                        
                    plt.plot(np.log10(all_train_loss),label='Train loss')
                    
                    plt.plot(np.log10(all_val_loss),label='Val loss')
                    plt.xlabel("Epoch")
                    plt.ylabel("MSE Loss")
                    # timestr = time.strftime("%Y%m%d-%H%M%S")
                    #plt.show()
                    plt.savefig(os.path.join(path_to_save_plots,'losses'))

            #writer.add_scalar('train_loss', all_train_loss[-1], global_step=i)
            #if len(all_val_loss)>0:
            #    writer.add_scalar('val_loss', all_val_loss[-1], global_step=i)
            #if args.lr_scheduler == 'ReduceLROnPlateau':
            #    writer.add_scalar('Epoch/learning_rate', optimizer.param_groups[0]['lr'], global_step=i)
            #elif args.lr_scheduler == 'CosineAnnealingLR':
            #    writer.add_scalar('Epoch/learning_rate', scheduler.get_last_lr()[0], global_step=i)

            


            end_i = time.time()
            # print(f"Epoch time: {(end_i-start_i)/60:.3f} seconds")

            
            model_state = {
                        'epoch': i + 1,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                }


           
            if Encoder is None:
                save_best_model(path_to_save_models, all_val_loss[-1], i, model_state, model, None, None, None)
            else:
                save_best_model(path_to_save_models, all_val_loss[-1], i, model_state, None, model, Encoder, Decoder)
           

            #lr_scheduler(loss_validation)

            early_stopping(all_val_loss[-1])
            if early_stopping.early_stop:
                break

        if args.support_tensors is True or args.support_test is True:
                del dummy_times
                
        end = time.time()
        # print(f"Training time: {(end-start)/60:.3f} minutes")
        # plt.figure()
        # plt.plot(np.linspace(0,len(Loss_print),len(Loss_print)),Loss_print)
        # plt.plot(np.linspace(0,len(Val_Loss),len(Val_Loss)),Val_Loss)
        # # plt.savefig('trained.png')
        # timestr = time.strftime("%Y%m%d-%H%M%S")
        # plt.savefig(os.path.join(path_to_save_plots,'trained'+timestr))
        # # plt.show()
        # plt.figure()
        # plt.plot(np.linspace(0,len(Loss_print),len(Loss_print)),np.log10(Loss_print))
        # plt.plot(np.linspace(0,len(Val_Loss),len(Val_Loss)),np.log10(Val_Loss))
        # timestr = time.strftime("%Y%m%d-%H%M%S")
        # plt.savefig(os.path.join(path_to_save_plots,'final_losses'+timestr))
        # # plt.show()
        
#     elif args.mode=='evaluate':
#         print('Running in evaluation mode')
#         ## Validating
#         model.eval()
        
#         t_min , t_max = args.time_interval
#         n_points = args.test_points

        
#         test_times=torch.sort(torch.rand(n_points),0)[0].to(device)*(t_max-t_min)+t_min
#         #test_times=torch.linspace(t_min,t_max,n_points)
        
#         #dummy_times = torch.cat([torch.Tensor([0.]).to(device),dummy_times])
#         # print('times :',times)
#         ###########################################################
        
#         with torch.no_grad():
                
#             model.eval()
#             if Encoder is not None:
#                 Encoder.eval()
#             if Decoder is not None:
#                 Decoder.eval()
                
#             test_loss = 0.0
#             loss_list = []
#             #counter = 0  

#             for obs_val, masks_val in tqdm(dataloaders['train']):
                
                



        
                
#                     obs_val = obs_val.to(args.device)
        
        
        
        
                                              
        
                    
#                     c= lambda x: \
#                     Encoder(obs_val[:,:,:,0,:].permute(0,3,1,2).requires_grad_(True))\
#                             .permute(0,2,3,1).unsqueeze(-2)\
#                             .contiguous().repeat(1,1,1,args.time_points,1).to(args.device)
                        
                    
        
                        
#                     y_0 =  Encoder(obs_val[:,:,:,0,:].permute(0,3,1,2))\
#                                 .permute(0,2,3,1).unsqueeze(-2)
#                     #print(c(1).shape)    
#                     if args.ts_integration is not None:
#                         times_integration = args.ts_integration
#                     else:
#                         times_integration = torch.linspace(0,1,args.time_points)
                    
#                     z_val = Integral_spatial_attention_solver_multbatch(
#                                         times_integration.to(args.device),
#                                         y_0.to(args.device),
#                                         y_init= None,
#                                         c=c,
#                                         sampling_points = args.time_points,
#                                         mask=mask,
#                                         Encoder = model,
#                                         max_iterations = args.max_iterations,
#                                         spatial_integration=True,
#                                         spatial_domain= spatial_domain.to(args.device),
#                                         spatial_domain_dim=2,
#                                         #lower_bound = lambda x: torch.Tensor([0]).to(device),
#                                         #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
#                                         smoothing_factor=args.smoothing_factor,
#                                         use_support=False,
#                                         accumulate_grads=True,
#                                         initialization=args.initialization
#                                         ).solve()
                    
                    
#                     z_val= Decoder(z_val.squeeze())

#                     z_val= z_val.permute(0,3,1,2)
#                     masks_val=masks_val.squeeze(-1)

#                     masks_val = masks_val.clone()
#                     masks_val[masks_val != 0] = 1
                
#                     test_loss += mse_error.item()
#                     loss_list.append(mse_error.item())
                
# #                 for in_batch_indx in range(args.n_batch):

# #                     obs_print = to_np(obs_test[in_batch_indx,:,:,:])
# #                     z_p = to_np(z_test[in_batch_indx,:,:,:])

# #                     #plot_reconstruction(obs_print, z_p, None, path_to_save_plots, 'plot_epoch_', i, args)
# #                     plot_reconstruction(obs_print, z_p, None, None, None, None, args)

# #                     plt.close('all')
# #                     del z_p, obs_print
#                     del z_test, obs_test
            
#             print(loss_list)
#             print("Average loss: ",test_loss*args.n_batch/obs.shape[0])
        
        #This was the original.

    elif args.mode == 'evaluate':
            print('Running in evaluation mode')
        
            test_loader = dataloaders['test']
        
            model.eval()
            if Encoder is not None:
                Encoder.eval()
            if Decoder is not None:
                Decoder.eval()
        
            all_pred = torch.tensor([]).to(args.device)
            all_labels = torch.tensor([]).to(args.device)
        
            with torch.no_grad():
                for obs_val, masks_val in tqdm(test_loader):
                    obs_val = obs_val.to(args.device)
                    masks_val = masks_val.to(args.device)
        
                    # Encoding input
                    c = lambda x: Encoder(
                        obs_val[:, :, :, 0, :].permute(0, 3, 1, 2).requires_grad_(True)
                    ).permute(0, 2, 3, 1).unsqueeze(-2).contiguous().repeat(1, 1, 1, args.time_points, 1).to(args.device)
        
                    y_0 = Encoder(
                        obs_val[:, :, :, 0, :].permute(0, 3, 1, 2)
                    ).permute(0, 2, 3, 1).unsqueeze(-2)
        
                    if args.ts_integration is not None:
                        times_integration = args.ts_integration
                    else:
                        times_integration = torch.linspace(0, 1, args.time_points).to(args.device)
        
                    # Solving dynamics
                    z_val = Integral_spatial_attention_solver_multbatch(
                        times_integration,
                        y_0.to(args.device),
                        y_init=None,
                        c=c,
                        sampling_points=args.time_points,
                        mask=mask,
                        Encoder=model,
                        max_iterations=args.max_iterations,
                        spatial_integration=True,
                        spatial_domain=spatial_domain.to(args.device),
                        spatial_domain_dim=2,
                        smoothing_factor=args.smoothing_factor,
                        use_support=False,
                        accumulate_grads=True,
                        initialization=args.initialization
                    ).solve()
        
                    # z_val = Decoder(z_val.squeeze())
                    # z_val = z_val.permute(0, 3, 1, 2)

                    # Reshape for new UNet decoder
                    z_decoder_input = z_val.squeeze(-2).permute(0, 3, 1, 2)
                    z_val = Decoder(z_decoder_input)  # → [B, 2, 256, 256]

                    
        
                    masks_val = masks_val.squeeze(-1).long()

        
                    preds = z_val.argmax(dim=1)
        
                    all_pred = torch.cat([all_pred, preds])
                    all_labels = torch.cat([all_labels, masks_val])

                        # Visualize 3 random predictions
                for i in range(100):
                    pred = all_pred[i].cpu().numpy()
                    label = all_labels[i].cpu().numpy()
                
                    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
                    axs[0].imshow(pred, cmap='gray')
                    axs[0].set_title("Predicted Mask")
                    axs[1].imshow(label, cmap='gray')
                    axs[1].set_title("Ground Truth")
                    plt.show()


    # Compute pixel-wise accuracy
    pixel_accuracy = (all_pred == all_labels).float().mean()

    print("Pixel-wise Accuracy:", pixel_accuracy.item())

    from sklearn.metrics import classification_report
    print(classification_report(
        to_np(all_labels.flatten()),
        to_np(all_pred.flatten()),
        # to_np(all_labels.flatten()),
        zero_division=0
    ))

    return pixel_accuracy


    # elif args.mode == 'evaluate':
    #     print('Running in evaluation mode')

    #         # ✅ Define save path
    #     str_model_name = "nie"
    #     str_log_dir = args.root_path
    #     path_to_experiment = os.path.join(str_log_dir, str_model_name, args.experiment_name)
        
    #     if args.resume_from_checkpoint:
    #         path_to_save_plots = os.path.join(path_to_experiment, args.resume_from_checkpoint, 'evaluation_plots')
    #     else:
    #         path_to_save_plots = os.path.join(path_to_experiment, 'evaluation_plots')
        
    #     if not os.path.exists(path_to_save_plots):
    #         os.makedirs(path_to_save_plots)
        
    #     print(f"Saving evaluation results to: {path_to_save_plots}")
    #     test_loader = dataloaders['test']
        
    #     model.eval()
    #     if Encoder is not None:
    #         Encoder.eval()
    #     if Decoder is not None:
    #         Decoder.eval()
        
    #     all_pred = []
    #     all_labels = []
    #     all_probs = []  # Store probabilities for confidence analysis
        
    #     with torch.no_grad():
    #         for obs_val, masks_val in tqdm(test_loader):
    #             obs_val = obs_val.to(args.device)
    #             masks_val = masks_val.to(args.device)
                
    #             # Encoding input
    #             c = lambda x: Encoder(
    #                 obs_val[:, :, :, 0, :].permute(0, 3, 1, 2).requires_grad_(True)
    #             ).permute(0, 2, 3, 1).unsqueeze(-2).contiguous().repeat(1, 1, 1, args.time_points, 1).to(args.device)
                
    #             y_0 = Encoder(
    #                 obs_val[:, :, :, 0, :].permute(0, 3, 1, 2)
    #             ).permute(0, 2, 3, 1).unsqueeze(-2)
                
    #             if args.ts_integration is not None:
    #                 times_integration = args.ts_integration
    #             else:
    #                 times_integration = torch.linspace(0, 1, args.time_points).to(args.device)
                
    #             # Solving dynamics
    #             z_val = Integral_spatial_attention_solver_multbatch(
    #                 times_integration,
    #                 y_0.to(args.device),
    #                 y_init=None,
    #                 c=c,
    #                 sampling_points=args.time_points,
    #                 mask=mask,
    #                 Encoder=model,
    #                 max_iterations=args.max_iterations,
    #                 spatial_integration=True,
    #                 spatial_domain=spatial_domain.to(args.device),
    #                 spatial_domain_dim=2,
    #                 smoothing_factor=args.smoothing_factor,
    #                 use_support=False,
    #                 accumulate_grads=True,
    #                 initialization=args.initialization
    #             ).solve()
                
    #             # ✅ NEW DECODER (correctly implemented)
    #             z_decoder_input = z_val.squeeze(-2).permute(0, 3, 1, 2)
    #             z_val = Decoder(z_decoder_input)  # → [B, 2, 256, 256]
                
    #             masks_val = masks_val.squeeze(-1).long()
                
    #             # Get predictions
    #             probs = torch.softmax(z_val, dim=1)  # [B, 2, H, W]
    #             preds = z_val.argmax(dim=1)  # [B, H, W]
                
    #             # Store for metrics (move to CPU to save GPU memory)
    #             all_pred.append(preds.cpu())
    #             all_labels.append(masks_val.cpu())
    #             all_probs.append(probs[:, 1].cpu())  # Crack class probability
        
    #     # Concatenate all batches
    #     all_pred = torch.cat(all_pred, dim=0)
    #     all_labels = torch.cat(all_labels, dim=0)
    #     all_probs = torch.cat(all_probs, dim=0)
        
    #     print("\n" + "="*70)
    #     print("EVALUATION RESULTS")
    #     print("="*70)
        
    #     # ========================================================================
    #     # 1. BASIC METRICS
    #     # ========================================================================
    #     from sklearn.metrics import classification_report, confusion_matrix
    #     import numpy as np
        
    #     pixel_accuracy = (all_pred == all_labels).float().mean()
    #     print(f"\n📊 Pixel-wise Accuracy: {pixel_accuracy.item():.4f}")
        
    #     # Classification report
    #     print("\n📋 Classification Report:")
    #     print(classification_report(
    #         to_np(all_labels.flatten()),
    #         to_np(all_pred.flatten()),
    #         target_names=['Background', 'Crack'],
    #         digits=4,
    #         zero_division=0
    #     ))
        
    #     # ========================================================================
    #     # 2. CONFUSION MATRIX (with visualization)
    #     # ========================================================================
    #     cm = confusion_matrix(
    #         to_np(all_labels.flatten()),
    #         to_np(all_pred.flatten())
    #     )
        
    #     print("\n📈 Confusion Matrix:")
    #     print(f"                 Predicted")
    #     print(f"              Background  Crack")
    #     print(f"Actual Background  {cm[0,0]:>8}  {cm[0,1]:>6}")
    #     print(f"       Crack       {cm[1,0]:>8}  {cm[1,1]:>6}")
        
    #     # ========================================================================
    #     # 3. CRACK-SPECIFIC METRICS
    #     # ========================================================================
    #     # True positives, false positives, false negatives
    #     tp = cm[1, 1]
    #     fp = cm[0, 1]
    #     fn = cm[1, 0]
    #     tn = cm[0, 0]
        
    #     iou_crack = tp / (tp + fp + fn + 1e-8)
    #     dice_crack = 2 * tp / (2 * tp + fp + fn + 1e-8)
        
    #     print(f"\n🎯 Crack Detection Metrics:")
    #     print(f"   IoU (Intersection over Union): {iou_crack:.4f}")
    #     print(f"   Dice Coefficient: {dice_crack:.4f}")
    #     print(f"   True Positives:  {tp:,} pixels")
    #     print(f"   False Positives: {fp:,} pixels")
    #     print(f"   False Negatives: {fn:,} pixels")
        
    #     # ========================================================================
    #     # 4. PREDICTION STATISTICS
    #     # ========================================================================
    #     total_predicted_crack = (all_pred == 1).sum().item()
    #     total_actual_crack = (all_labels == 1).sum().item()
    #     total_pixels = all_pred.numel()
        
    #     print(f"\n📊 Prediction Statistics:")
    #     print(f"   Total pixels: {total_pixels:,}")
    #     print(f"   Actual crack pixels: {total_actual_crack:,} ({100*total_actual_crack/total_pixels:.2f}%)")
    #     print(f"   Predicted crack pixels: {total_predicted_crack:,} ({100*total_predicted_crack/total_pixels:.2f}%)")
    #     print(f"   Prediction ratio: {total_predicted_crack/total_actual_crack:.2f}x actual")
        
    #     # ========================================================================
    #     # 5. CONFIDENCE ANALYSIS
    #     # ========================================================================
    #     crack_probs = all_probs[all_labels == 1]  # Probabilities where cracks exist
    #     bg_probs = all_probs[all_labels == 0]     # Probabilities where no cracks
        
    #     print(f"\n🎲 Model Confidence:")
    #     print(f"   Avg confidence on crack pixels: {crack_probs.mean():.4f}")
    #     print(f"   Avg confidence on background: {1-bg_probs.mean():.4f}")
    #     print(f"   Min crack confidence: {crack_probs.min():.4f}")
    #     print(f"   Max crack confidence: {crack_probs.max():.4f}")
        
    #     # ========================================================================
    #     # 6. VISUALIZATIONS
    #     # ========================================================================
    #     import matplotlib.pyplot as plt
        
    #     print(f"\n🖼️  Generating visualizations...")
        
    #     # Select diverse examples: best, worst, and random
    #     sample_indices = []
        
    #     # Calculate per-image IoU to find best/worst
    #     per_image_iou = []
    #     for i in range(len(all_pred)):
    #         pred_i = all_pred[i]
    #         label_i = all_labels[i]
            
    #         tp_i = ((pred_i == 1) & (label_i == 1)).sum().item()
    #         fp_i = ((pred_i == 1) & (label_i == 0)).sum().item()
    #         fn_i = ((pred_i == 0) & (label_i == 1)).sum().item()
            
    #         iou_i = tp_i / (tp_i + fp_i + fn_i + 1e-8)
    #         per_image_iou.append(iou_i)
        
    #     per_image_iou = np.array(per_image_iou)
        
    #     # Select examples
    #     best_idx = np.argmax(per_image_iou)
    #     worst_idx = np.argmin(per_image_iou)
    #     median_idx = np.argsort(per_image_iou)[len(per_image_iou)//2]
    #     random_indices = np.random.choice(len(all_pred), size=2, replace=False)
        
    #     sample_indices = [best_idx, worst_idx, median_idx] + random_indices.tolist()
        
    #     # Visualize selected samples
    #     n_samples = len(sample_indices)
    #     fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))
        
    #     if n_samples == 1:
    #         axes = axes[np.newaxis, :]
        
    #     for idx, sample_idx in enumerate(sample_indices):
    #         pred = all_pred[sample_idx].numpy()
    #         label = all_labels[sample_idx].numpy()
    #         prob = all_probs[sample_idx].numpy()
            
    #         # Compute overlay (errors)
    #         # Green: True positive, Red: False positive, Blue: False negative
    #         overlay = np.zeros((*pred.shape, 3))
    #         overlay[..., 1] = (pred == 1) & (label == 1)  # TP - Green
    #         overlay[..., 0] = (pred == 1) & (label == 0)  # FP - Red
    #         overlay[..., 2] = (pred == 0) & (label == 1)  # FN - Blue
            
    #         # Plot
    #         axes[idx, 0].imshow(label, cmap='gray')
    #         axes[idx, 0].set_title(f"Ground Truth (IoU: {per_image_iou[sample_idx]:.3f})")
    #         axes[idx, 0].axis('off')
            
    #         axes[idx, 1].imshow(pred, cmap='gray')
    #         axes[idx, 1].set_title("Prediction")
    #         axes[idx, 1].axis('off')
            
    #         axes[idx, 2].imshow(overlay)
    #         axes[idx, 2].set_title("Overlay (G=TP, R=FP, B=FN)")
    #         axes[idx, 2].axis('off')
        
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(path_to_save_plots, 'evaluation_results.png'), dpi=150, bbox_inches='tight')
    #     plt.show()
        
    #     print(f"✓ Visualizations saved to: {os.path.join(path_to_save_plots, 'evaluation_results.png')}")
        
    #     # ========================================================================
    #     # 7. THICKNESS ANALYSIS (Detect blob problem)
    #     # ========================================================================
    #     from scipy.ndimage import distance_transform_edt
        
    #     print(f"\n📏 Analyzing prediction thickness...")
        
    #     # Sample a few images to analyze thickness
    #     thickness_samples = []
    #     for i in range(min(20, len(all_pred))):
    #         pred_i = all_pred[i].numpy()
    #         label_i = all_labels[i].numpy()
            
    #         if label_i.sum() > 100:  # Only analyze images with cracks
    #             # Distance transform gives distance to nearest background pixel
    #             pred_dist = distance_transform_edt(pred_i)
    #             label_dist = distance_transform_edt(label_i)
                
    #             # Average "thickness" (max distance from edge)
    #             pred_thickness = pred_dist.max()
    #             label_thickness = label_dist.max()
                
    #             thickness_samples.append({
    #                 'pred': pred_thickness,
    #                 'label': label_thickness,
    #                 'ratio': pred_thickness / (label_thickness + 1e-8)
    #             })
        
    #     if thickness_samples:
    #         avg_pred_thickness = np.mean([s['pred'] for s in thickness_samples])
    #         avg_label_thickness = np.mean([s['label'] for s in thickness_samples])
    #         avg_ratio = np.mean([s['ratio'] for s in thickness_samples])
            
    #         print(f"   Avg ground truth thickness: {avg_label_thickness:.2f} pixels")
    #         print(f"   Avg predicted thickness: {avg_pred_thickness:.2f} pixels")
    #         print(f"   Thickness ratio (pred/truth): {avg_ratio:.2f}x")
            
    #         if avg_ratio > 2.0:
    #             print(f"   ⚠️  WARNING: Predictions are too thick! (>2x ground truth)")
    #         elif avg_ratio < 0.5:
    #             print(f"   ⚠️  WARNING: Predictions are too thin! (<0.5x ground truth)")
    #         else:
    #             print(f"   ✓ Thickness looks reasonable")
        
    #     print("\n" + "="*70)
    #     print("EVALUATION COMPLETE")
    #     print("="*70)
        
    #     return {
    #         'pixel_accuracy': pixel_accuracy.item(),
    #         'iou': iou_crack,
    #         'dice': dice_crack,
    #         'precision': tp / (tp + fp + 1e-8),
    #         'recall': tp / (tp + fn + 1e-8),
    #         'f1_score': dice_crack,  # Dice = F1 for binary segmentation
    #     }


        


    
        
    
        
