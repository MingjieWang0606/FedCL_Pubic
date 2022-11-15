import torch
import torch.nn.functional as F
import numpy as np
import random
import heapq
from random import choice
from sklearn.model_selection import train_test_split
from FLAlgorithms.users.userbase import User
from FLAlgorithms.curriculum.cl_score import CL_User_Score
from FLAlgorithms.trainmodel.generator import Discriminator
import pdb
import decimal
def median(x):
    x = sorted(x)
    length = len(x)
    mid, rem = divmod(length, 2)
    if rem:
        return x[:mid], x[mid+1:], x[mid]
    else:
        return x[:mid], x[mid:], x[mid-1]
    
    
    
class UserpFedGen(User):
    def __init__(self,
                 args, id, model, generative_model,
                 train_data, test_data,
                 available_labels, latent_layer_idx, label_info,
                 use_adam=False):
        super().__init__(args, id, model, train_data, test_data, use_adam=use_adam)
        self.gen_batch_size = args.gen_batch_size
        self.generative_model = generative_model
        self.Discriminator = Discriminator()
        self.latent_layer_idx = latent_layer_idx
        self.available_labels = available_labels
        self.label_info=label_info
        self.CL_User_Score=CL_User_Score

        self.optimizer_discriminator = torch.optim.Adam(
            params=self.model.parameters(),
            lr=1e-4, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=0, amsgrad=False)
        self.lr_scheduler_discriminator = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer_discriminator, gamma=0.98)
        
        
    def exp_lr_scheduler(self, epoch, decay=0.98, init_lr=0.1, lr_decay_epoch=1):
        """Decay learning rate by a factor of 0.95 every lr_decay_epoch epochs."""
        lr= max(1e-4, init_lr * (decay ** (epoch // lr_decay_epoch)))
        return lr

    def update_label_counts(self, labels, counts):
        for label, count in zip(labels, counts):
            self.label_counts[int(label)] += count

    def clean_up_counts(self):
        del self.label_counts
        self.label_counts = {label:1 for label in range(self.unique_labels)}

    def train(self, glob_iter, personalized=False, early_stop=100, regularization=True, verbose=False, run_curriculum=True,cl_score_norm_list=None,server_epoch=None,gmm_=None,gmm_len=None,next_stage=None):
        self.clean_up_counts()
        self.model.train()
        self.generative_model.eval()
        part_loss =0
        TEACHER_LOSS, DIST_LOSS, LATENT_LOSS, CL_SCORE = 0, 0, 0, 1
        exit_ = False
        for epoch in range(self.local_epochs):
            self.model.train()
            #print('epoch:'+str( epoch))
            #for i in range(self.K):
            self.optimizer.zero_grad()
            #### sample from real dataset (un-weighted)
            samples =self.get_next_train_batch(count_labels=True)
            X, y = samples['X'], samples['y']
            self.update_label_counts(samples['labels'], samples['counts'])
            model_result=self.model(X, logit=True)
            user_output_logp_ = model_result['output']
            user_output_logp = model_result['output']
            CL_results = self.CL_User_Score(model_result = model_result, 
                                                 Algorithms = 'SuperLoss_ce', 
                                                 loss_fun = self.loss,
                                                 y = y,
                                                 local_epoch = epoch,
                                                 schedule = [epoch,self.local_epochs]
                                                )
            predictive_loss=CL_results['Loss']
            CL_Results_Score = CL_results['Curriculum_Learning_Score'] 

            #print(CL_results['score_list'])
            #print(CL_results['celoss'])
            # -----------------
            #  Train Generator
            # -----------------
            #### sample y and generate z
            if regularization and epoch < early_stop:

                generative_alpha=self.exp_lr_scheduler(glob_iter, decay=0.98, init_lr=self.generative_alpha)
                generative_beta=self.exp_lr_scheduler(glob_iter, decay=0.98, init_lr=self.generative_beta)
                ### get generator output(latent representation) of the same label  
                gmm_results,_ = gmm_.sample(1)
                
                real_cl_score_ = torch.tensor(gmm_results, dtype=torch.float)[:y.size()[0]].view(y.size()[0],1)
                

                #real_cl_score_ =torch.tensor(CL_results_score_list, dtype=torch.float).view(y.size()[0],1)

                gen_output_=self.generative_model(y, real_cl_score_, latent_layer_idx=self.latent_layer_idx)
                gen_output = gen_output_['output'].clone().detach()
                logit_given_gen=self.model(gen_output, start_layer_idx=self.latent_layer_idx, logit=True)['logit']
                target_p=F.softmax(logit_given_gen, dim=1).clone().detach()
                user_latent_loss= generative_beta * self.ensemble_loss(user_output_logp, target_p)

                sampled_y=np.random.choice(self.available_labels, self.gen_batch_size)

                sampled_y=torch.tensor(sampled_y)
                #CL_results_score_list_norm = (CL_results_score_list-min(cl_score_norm_list)) / (max(cl_score_norm_list) - min(cl_score_norm_list))
                #CL_results_score_list_norm = cl_score_norm_list
                
                l_Half, r_Half, q2 = median(gmm_.sample(gmm_len)[0].flatten())
                lHalf = median(l_Half)[2]
                rHalf = median(r_Half)[2]
                #50,100,150,200
                if next_stage==0:
                    cl_score_fake_=[random.uniform(0,lHalf) for i in range(self.gen_batch_size)]
                elif next_stage==1:
                    cl_score_fake_=[random.uniform(lHalf,rHalf) for i in range(self.gen_batch_size)]
                elif next_stage>=2:
                    cl_score_fake_=[random.uniform(rHalf,1) for i in range(self.gen_batch_size)]

                #$print(cl_score_fake_)
#                     easy_number = int(self.gen_batch_size/4*3)
#                     hard_number = int(self.gen_batch_size - easy_number)
#                     cl_score_fake_easy=[random.uniform(0,max(CL_results['score_list'])) for i in range(easy_number)]
#                     cl_score_fake_hard=[random.uniform(min(CL_results['score_list']),0) for i in range(hard_number)]
#                     cl_score_fake_ = cl_score_fake_easy + cl_score_fake_hard
                random.shuffle(cl_score_fake_)
                #print(float(CL_results_score_list[(CL_results_score_list_norm == lHalf).nonzero()[0][0]]))
                cl_score_fake_=torch.tensor(cl_score_fake_, dtype=torch.float).view(self.gen_batch_size,1)
                gen_result=self.generative_model(sampled_y,cl_score_fake_, latent_layer_idx=self.latent_layer_idx)


                gen_output=gen_result['output'] # latent representation when latent = True, x otherwise

                user_output_logp = self.model(gen_output, start_layer_idx=self.latent_layer_idx, logit=True)
                CL_score_fake_results = self.CL_User_Score(model_result = user_output_logp, 
                                                 Algorithms = 'SuperLoss_ce', 
                                                 loss_fun = self.loss,
                                                 y = sampled_y,
                                                 local_epoch = epoch,
                                                 schedule = [epoch,self.local_epochs]
                                                )['score_list_base']

                values_ = ((CL_score_fake_results-min(cl_score_norm_list)) / (max(cl_score_norm_list) - min(cl_score_norm_list)))


                #print(CL_results_score_list[(CL_results_score_list_norm == lHalf).nonzero()[0][0]])
                #print(CL_score_fake_results)
#                     print(sorted( values_,reverse=False))
#                     print(sorted( values_,reverse=False)[int(len(values_)/1.1)])
#                     exit()
                ##print(sorted( values_,reverse=False)[int(len(values_)*0.6)])
                #print(CL_results_score_list[(CL_results_score_list_norm == lHalf).nonzero()[0][0]])
#                 if next_stage:
#                     if sorted( values_,reverse=False)[int(len(values_)*0.8)] < lHalf:
#                         print(epoch)
#                         break
#                 else:
#                     if sorted( values_,reverse=True)[int(len(values_)*0.8)] > lHalf:
#                         print(epoch)
#                         break
                        
                if next_stage==0:
                    if sorted(values_,reverse=False)[int(len(values_)*0.8)] < lHalf:
                        print(epoch)
                        exit_ = True
                        break
                elif next_stage==1:
                    if  rHalf  > sorted(values_,reverse=False)[int(len(values_)*0.8)] > lHalf :
                        print(epoch)
                        exit_ = True
                        break
                else:
                    if sorted( values_,reverse=False)[int(len(values_)*0.8)] > rHalf:
                        print(epoch)
                        exit_ = True
                        break    
                user_output_logp = user_output_logp['output']

                teacher_loss =  generative_alpha * torch.mean(
                    self.generative_model.crossentropy_loss(user_output_logp, sampled_y)
                )
                # this is to further balance oversampled down-sampled synthetic data
                gen_ratio = self.gen_batch_size / self.batch_size
                loss=predictive_loss + gen_ratio * teacher_loss + user_latent_loss
                TEACHER_LOSS+=teacher_loss
                LATENT_LOSS+=user_latent_loss

#                     if all( value < CL_results_score_list[(CL_results_score_list_norm == lHalf).nonzero()[0][0]] for value in sorted( values_,reverse=False)[:int(len(values_)/1.2)]):

#                         break
            else:
                #### get loss and perform optimization
                
                loss=predictive_loss
            loss.backward(retain_graph=True)
            self.optimizer.step()#self.local_model)

            # ---------------------
            #  Train Discriminator
            # --------------------- 

            if regularization and epoch < early_stop:
                pass
#                     self.optimizer_discriminator.zero_grad()
#                     print(X.size())
#                     print(gen_output_['output'].size())
#                     D_H_real_loss = self.dist_loss(X, torch.ones_like(user_output_logp_))
#                     D_H_fake_loss = self.dist_loss(gen_output_, torch.zeros_like(gen_output))       


    
        # local-model <=== self.model
        self.clone_model_paramenter(self.model.parameters(), self.local_model)
        if personalized:
            self.clone_model_paramenter(self.model.parameters(), self.personalized_model_bar)
        self.lr_scheduler.step(glob_iter)
        
        if regularization and verbose:
            try:
                TEACHER_LOSS=TEACHER_LOSS.detach().numpy() / (self.local_epochs * self.K)
            except:
                TEACHER_LOSS=TEACHER_LOSS / (self.local_epochs * self.K)
            try:
                LATENT_LOSS=LATENT_LOSS.detach().numpy() / (self.local_epochs * self.K)
            except:
                LATENT_LOSS=LATENT_LOSS / (self.local_epochs * self.K)
            info='\nUser Teacher Loss={:.4f}'.format(TEACHER_LOSS)
            info+=', Latent Loss={:.4f}'.format(LATENT_LOSS)
            print(info) 

        if CL_Results_Score!= None:
            return CL_results['score_list'],part_loss,exit_
        else:
            return None
        
    def adjust_weights(self, samples):
        labels, counts = samples['labels'], samples['counts']
        #weight=self.label_weights[y][:, user_idx].reshape(-1, 1)
        np_y = samples['y'].detach().numpy()
        n_labels = samples['y'].shape[0]
        weights = np.array([n_labels / count for count in counts]) # smaller count --> larger weight
        weights = len(self.available_labels) * weights / np.sum(weights) # normalized
        label_weights = np.ones(self.unique_labels)
        label_weights[labels] = weights
        sample_weights = label_weights[np_y]
        return sample_weights
    

