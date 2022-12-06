from FLAlgorithms.users.userpFedGen import UserpFedGen
from FLAlgorithms.servers.serverbase import Server
from FLAlgorithms.curriculum.cl_score import CL_User_Score
from utils.model_utils import read_data, read_user_data, aggregate_user_data, create_generative_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
import os
import copy
import time
import random
from sklearn.mixture import GaussianMixture
MIN_SAMPLES_PER_LABEL=1

class FedCL(Server):
    def __init__(self, args, model, seed):
        super().__init__(args, model, seed)

        # Initialize data for all users
        data = read_data(args.dataset)
        # data contains: clients, groups, train_data, test_data, proxy_data
        clients = data[0]
        total_users = len(clients)
        self.total_test_samples = 0
        self.local = 'local' in self.algorithm.lower()
        self.use_adam = 'adam' in self.algorithm.lower()

        self.early_stop = 20  # stop using generated samples after 20 local epochs
        self.student_model = copy.deepcopy(self.model)
        self.generative_model = create_generative_model(args.dataset, args.algorithm, self.model_name, args.embedding)
        if not args.train:
            print('number of generator parameteres: [{}]'.format(self.generative_model.get_number_of_parameters()))
            print('number of model parameteres: [{}]'.format(self.model.get_number_of_parameters()))
        self.latent_layer_idx = self.generative_model.latent_layer_idx
        self.init_ensemble_configs()
        print("latent_layer_idx: {}".format(self.latent_layer_idx))
        print("label embedding {}".format(self.generative_model.embedding))
        print("ensemeble learning rate: {}".format(self.ensemble_lr))
        print("ensemeble alpha = {}, beta = {}, eta = {}".format(self.ensemble_alpha, self.ensemble_beta, self.ensemble_eta))
        print("generator alpha = {}, beta = {}".format(self.generative_alpha, self.generative_beta))
        self.init_loss_fn()
        self.train_data_loader, self.train_iter, self.available_labels = aggregate_user_data(data, args.dataset, self.ensemble_batch_size)
        self.generative_optimizer = torch.optim.Adam(
            params=self.generative_model.parameters(),
            lr=self.ensemble_lr, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=self.weight_decay, amsgrad=False)
        self.generative_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.generative_optimizer, gamma=0.98)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.ensemble_lr, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=0, amsgrad=False)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.98)
        self.CL_User_Score=CL_User_Score
        
        
        #### creating users ####
        self.users = []
        for i in range(total_users):
            id, train_data, test_data, label_info =read_user_data(i, data, dataset=args.dataset, count_labels=True)
            self.total_train_samples+=len(train_data)
            self.total_test_samples += len(test_data)
            id, train, test=read_user_data(i, data, dataset=args.dataset)
            user=UserpFedGen(
                args, id, model, self.generative_model,
                train_data, test_data,
                self.available_labels, self.latent_layer_idx, label_info,
                use_adam=self.use_adam)
            self.users.append(user)
        print("Number of Train/Test samples:", self.total_train_samples, self.total_test_samples)
        print("Data from {} users in total.".format(total_users))
        print("Finished creating FedAvg server.")
# 
    def train(self, args):
        #### pretraining
        best_auc = -np.inf
        cl_score_norm_list_1 = []
        cl_score_norm_list_2 = []
        
        CL_Results_Score_list_1 = []
        CL_Results_Score_list_2 = []
        gmm_cnn = None
        _ = None
        next_stage__ = 0
        for glob_iter in range(self.num_glob_iters):
            print("\n\n-------------Round number: ",glob_iter, " -------------\n\n")
            self.selected_users, self.user_idxs=self.select_users(glob_iter, self.num_users, return_idx=True)
            if not self.local:
                self.send_parameters(mode=self.mode)# broadcast averaged prediction model
            self.evaluate()
            chosen_verbose_user = np.random.randint(0, len(self.users))
            self.timestamp = time.time() # log user-training start time
            
            loss_list = []
            # Train model base on local
            if  glob_iter > 0:
                cl_score_norm_list_1 = []
            if len(cl_score_norm_list_2)>1:
                print(len(np.array(cl_score_norm_list_2).reshape(-1)))
                _ = np.array(cl_score_norm_list_2).reshape(len(self.selected_users), -1)
                gmm_cnn=GaussianMixture(n_components=len(self.selected_users), covariance_type="spherical", random_state=0)
                gmm_cnn.fit(_)
                
            if  glob_iter == 0:    
                Break_local_list = [False for i in range(len(self.user_idxs))]        
                
            for i,(user_id, user) in enumerate(zip(self.user_idxs, self.selected_users)):
                verbose = user_id == chosen_verbose_user
                # perform regularization using generated samples after the first communication round
                
                CL_Results_Score, loss_, Break_local = user.train(
                    glob_iter,
                    personalized=self.personalized,
                    early_stop=self.early_stop,
                    verbose=verbose and glob_iter > 0,
                    regularization= glob_iter > 0 ,
                    run_curriculum = True,
                    cl_score_norm_list = cl_score_norm_list_2,
                    server_epoch = glob_iter,
                    gmm_ = gmm_cnn if gmm_cnn != None else None,
                    gmm_len = 320,
                    next_stage = next_stage__
                
                )
                CL_Results_Score_list_1.append([CL_Results_Score.clone().detach().numpy()])
                
                cl_score_norm_list_1.extend(CL_Results_Score)
                    
                loss_list.append(loss_)
                if Break_local and glob_iter > 2:
                        Break_local_list[i] = True
#             print('*'*20)
#             print(Break_local_list.count(True))
#             print('*'*20)
            if Break_local_list.count(True)>=len(self.user_idxs)*0.8:
                next_stage__ += 1
                Break_local_list = [False for i in range(len(self.user_idxs))]     
                print('*'*20)
                print(glob_iter)
                print('*'*20)
                
            cl_score_norm_list_2 = cl_score_norm_list_1
            CL_Results_Score_list_2 = CL_Results_Score_list_1
            
            #import pdb; pdb.set_trace()
            curr_timestamp = time.time() # log  user-training end time
            train_time = (curr_timestamp - self.timestamp) / len(self.selected_users)
            self.metrics['user_train_time'].append(train_time)
            if self.personalized:
                self.evaluate_personalized_model()

            self.timestamp = time.time() # log server-agg start time
            self.train_generator(
                self.batch_size,
                epoches=self.ensemble_epochs // self.n_teacher_iters,
                latent_layer_idx=self.latent_layer_idx,
                verbose=True,
                Real_CL_Results = CL_Results_Score_list_2
            )
            self.aggregate_parameters()
            curr_timestamp=time.time()  # log  server-agg end time
            agg_time = curr_timestamp - self.timestamp
            self.metrics['server_agg_time'].append(agg_time)
            if glob_iter  > 0 and glob_iter % 20 == 0 and self.latent_layer_idx == 0:
                self.visualize_images(self.generative_model, glob_iter, repeats=10)

        self.save_results(args)
        self.save_model()

    def train_generator(self, batch_size, epoches=1, latent_layer_idx=-1, verbose=False, Real_CL_Results=None,Real_CL_Results_sum=None):
        """
        Learn a generator that find a consensus latent representation z, given a label 'y'.
        :param batch_size:
        :param epoches:
        :param latent_layer_idx: if set to -1 (-2), get latent representation of the last (or 2nd to last) layer.
        :param verbose: print loss information.
        :return: Do not return anything.
        """
        #self.generative_regularizer.train()
        self.label_weights, self.qualified_labels = self.get_label_weights()
        TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS, STUDENT_LOSS2 = 0, 0, 0, 0

        def update_generator_(n_iters, student_model, TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS,Real_CL_Results):
            self.generative_model.train()
            student_model.eval()
            for i in range(n_iters):
                self.generative_optimizer.zero_grad()
                y=np.random.choice(self.qualified_labels, batch_size)
                y_input=torch.LongTensor(y)
                
                ## feed to generator
                Real_CL_Results = np.squeeze(np.array(Real_CL_Results))
                gmm=GaussianMixture(n_components=len(self.selected_users), covariance_type="spherical", random_state=0)
                gmm.fit(Real_CL_Results)
                gmm_,_ = gmm.sample(1)
                
                cl_sample = torch.tensor(gmm_, dtype=torch.float).view(batch_size,1)
                
                #torch.tensor([random.uniform(min(Real_CL_Results[random.randint(0,len(self.selected_users)-1)][0]), max(Real_CL_Results[random.randint(0,len(self.selected_users)-1)][0])) for i in range(batch_size)], dtype=torch.float).view(batch_size,1)
                
                gen_result=self.generative_model(y_input,cl_sample, latent_layer_idx=latent_layer_idx, verbose=True)

                # get approximation of Z( latent) if latent set to True, X( raw image) otherwise
                gen_output, eps=gen_result['output'], gen_result['eps']

                ##### get losses ####x
                # decoded = self.generative_regularizeen_output)
                # regularization_loss = beta * self.generative_model.dist_loss(decoded, eps) # map generated z back to eps
                diversity_loss=self.generative_model.diversity_loss(eps, gen_output) 

                ######### get teacher loss ############
                teacher_loss=0
                # teacher_logit=0
                fake_cl_score = 0
                diversity_loss_list = 0
                
                for user_idx, user in enumerate(self.selected_users):
                    gmm_,_ = gmm.sample(1)
                    cl_sample = torch.tensor(gmm_, dtype=torch.float).view(batch_size,1)
    #torch.tensor([random.uniform(min(Real_CL_Results[user_idx][0]), max(Real_CL_Results[user_idx][0])) for i in range(batch_size)], dtype=torch.float).view(batch_size,1)   
                    gen_result=self.generative_model(y_input,cl_sample, latent_layer_idx=latent_layer_idx, verbose=True)

                # get approximation of Z( latent) if latent set to True, X( raw image) otherwise
                    gen_output, eps=gen_result['output'], gen_result['eps']
                    user.model.eval()
                    weight=self.label_weights[y][:, user_idx].reshape(-1, 1)
                    expand_weight=np.tile(weight, (1, self.unique_labels))
                    user_result_given_gen=user.model(gen_output, start_layer_idx=latent_layer_idx, logit=True)
                    
                    CL_results = self.CL_User_Score(model_result = user_result_given_gen, 
                                                     Algorithms = 'SuperLoss_ce', 
                                                     loss_fun = None,
                                                     y = y_input,
                                                     local_epoch = None,
                                                     schedule = None,
                                                    
                                                    )
 
                    user_output_logp_=F.log_softmax(user_result_given_gen['logit'], dim=1).squeeze(-1)
                    user_output_logp_ = {'logit':user_output_logp_}
                    fake_cl_loss_ = self.CL_User_Score(model_result = user_output_logp_, 
                                                     Algorithms = 'SuperLoss_ce', 
                                                     loss_fun = None,
                                                     y = y_input,
                                                     local_epoch = None,
                                                     schedule = None
                                                    )
                    teacher_loss_=torch.mean( \
                        fake_cl_loss_['Loss']* \
                        torch.tensor(weight, dtype=torch.float32))
                    
                    cl_loss = torch.mean(torch.nn.MSELoss(reduce=False, size_average=False)(torch.tensor(CL_results['score_list'].clone().detach(), dtype=torch.float),torch.tensor(fake_cl_loss_['score_list'].clone().detach(), dtype=torch.float)))
                        
                    teacher_loss += teacher_loss_
                    fake_cl_score += cl_loss
                    #teacher_logit+=user_result_given_gen['logit'] * torch.tensor(expand_weight, dtype=torch.float32)
                
                teacher_loss = teacher_loss#/len(self.selected_users)

                student_loss=1
                if self.ensemble_beta > 0:
                    loss=self.ensemble_alpha * teacher_loss - self.ensemble_beta * student_loss + self.ensemble_eta * diversity_loss# + fake_cl_score
                else:
                    loss=self.ensemble_alpha * teacher_loss + self.ensemble_eta * diversity_loss# + fake_cl_score
                    
                loss.backward()
                self.generative_optimizer.step()
                TEACHER_LOSS += self.ensemble_alpha * teacher_loss#(torch.mean(TEACHER_LOSS.double())).item()
                STUDENT_LOSS += self.ensemble_beta * student_loss#(torch.mean(student_loss.double())).item()
                DIVERSITY_LOSS += self.ensemble_eta * diversity_loss#(torch.mean(diversity_loss.double())).item()
                
            return TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS,fake_cl_score#,Triplet_Loss

        for i in range(epoches):
            TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS, fake_cl_score=update_generator_(
                self.n_teacher_iters, self.model, TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS,Real_CL_Results)

        TEACHER_LOSS = TEACHER_LOSS.detach().numpy() / (self.n_teacher_iters * epoches)
        STUDENT_LOSS = STUDENT_LOSS/ (self.n_teacher_iters * epoches)
        DIVERSITY_LOSS = DIVERSITY_LOSS.detach().numpy() / (self.n_teacher_iters * epoches)
        fake_cl_score = fake_cl_score.detach().numpy() / (self.n_teacher_iters * epoches)
        #Triplet_Loss_ = Triplet_Loss_.detach().numpy() / (self.n_teacher_iters * epoches)
        info="Generator: Teacher Loss= {:.4f}, Student Loss= {:.4f}, Diversity Loss = {:.4f}, CL Loss = {:.4f},". \
            format(TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS,fake_cl_score)
        if verbose:
            print(info)
        self.generative_lr_scheduler.step()


    def get_label_weights(self):
        label_weights = []
        qualified_labels = []
        for label in range(self.unique_labels):
            weights = []
            for user in self.selected_users:
                weights.append(user.label_counts[label])
            if np.max(weights) > MIN_SAMPLES_PER_LABEL:
                qualified_labels.append(label)
            # uniform
            label_weights.append( np.array(weights) / np.sum(weights) )
        label_weights = np.array(label_weights).reshape((self.unique_labels, -1))
        return label_weights, qualified_labels

    def visualize_images(self, generator, glob_iter, repeats=1):
        """
        Generate and visualize data for a generator.
        """
        os.system("mkdir -p images")
        path = f'images/{self.algorithm}-{self.dataset}-iter{glob_iter}.png'
        y=self.available_labels
        y = np.repeat(y, repeats=repeats, axis=0)
        y_input=torch.tensor(y)
        generator.eval()
        images=generator(y_input, latent=False)['output'] # 0,1,..,K, 0,1,...,K
        images=images.view(repeats, -1, *images.shape[1:])
        images=images.view(-1, *images.shape[2:])
        save_image(images.detach(), path, nrow=repeats, normalize=True)
        print("Image saved to {}".format(path))
