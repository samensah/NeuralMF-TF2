import numpy as np
from absl import logging
import time
from model import NeuralMF
from loader import return_data
from pathlib import Path
from tensorflow import keras
import tensorflow as tf

class Solver(object):
    def __init__(self, opt):
        self.opt = opt
        self.epoch = opt['epoch']
        self.batch_size = opt['batch_size']
        self.lr = opt['lr']
        self.K = opt['K']
        self.global_epoch = 0
        self.checkpoints_to_keep = 3
        self.log_file = opt['log_file']

        # Dataset
        self.data_loader = dict()
        self.data_loader['train'], self.data_loader['test'], user_num, item_num, train_num, test_num = return_data(opt)
        self.opt['user_num'] = user_num
        self.opt['item_num'] = item_num
        self.opt['train_num'] = train_num
        self.opt['test_num'] = test_num
        
        # Network & Optimizer
        self.model = NeuralMF(self.opt)
        self.optim  = keras.optimizers.Adam(lr=self.lr)

        self.ckpt_dir = Path(opt['ckpt_dir'])
        if not self.ckpt_dir.exists() : self.ckpt_dir.mkdir(parents=True,exist_ok=True)
        self.load_ckpt = opt['load_ckpt']
        #self.load_ckpt = self.load_checkpoint('checkpoints\\ckpt-560')
        if self.load_ckpt != '' : self.load_checkpoint(self.load_ckpt)

        # loss function
        self.mse_lossfn = keras.losses.MeanSquaredError(reduction=keras.losses.Reduction.SUM)
        self.mae_lossfn = keras.losses.MeanAbsoluteError(reduction= keras.losses.Reduction.SUM)
        
        # History
        self.history = dict()
        self.history['test_RMSE'] = []
        self.history['test_MAE'] = []


    def train(self):
        for e in range(self.epoch):
            self.global_epoch += 1
            for batch in self.data_loader['train']:
                user_id, item_id, label = batch[:,0], batch[:,1], batch[:,2]
                with tf.GradientTape() as tape:
                    logit = self.model((user_id, item_id), training=True)
                    main_loss = tf.reduce_mean(self.mse_lossfn(logit, label))
                    total_loss = tf.add_n([main_loss] + self.model.losses) # add model regularization loss
                gradients = tape.gradient(total_loss, self.model.trainable_variables)
                self.optim.apply_gradients(zip(gradients, self.model.trainable_variables))
            
            # evaluation after every epoch    
            self.test()

        print(" [*] Training Finished!")
        best_rmse = min(self.history['test_RMSE'])
        best_index = self.history['test_RMSE'].index(best_rmse)
        best_mae = self.history['test_MAE'][best_index]
        print("[*] best rmse and mae:{:.4f}, {:.4f}".format(best_rmse, best_mae))

    def test(self, save_ckpt=True):
        total_mse_loss, total_mae_loss = 0., 0. 
        for batch in self.data_loader['test']:
            user_id, item_id, label = batch[:,0], batch[:,1], batch[:,2]
            logit = self.model((user_id, item_id), training=False)

            total_mse_loss += self.mse_lossfn(logit, label).numpy()
            total_mae_loss += self.mae_lossfn(logit, label).numpy()

        total_mse_loss = total_mse_loss / self.opt['test_num'] 
        total_rmse_loss = np.sqrt(total_mse_loss)
        total_mae_loss = total_mae_loss / self.opt['test_num'] 

        self.history['test_RMSE'].append(total_rmse_loss)
        self.history['test_MAE'].append(total_mae_loss)
        print('[epoch{}] test rmse: {}, mae: {}'.format(self.global_epoch, total_rmse_loss, total_mae_loss))

        log_path = Path.cwd() / "log"
        if not log_path.exists(): log_path.mkdir(parents=True, exist_ok=True)
        open(log_path / self.log_file, 'a').write('[epoch{}] test rmse: {}, mae: {}\n'.format(self.global_epoch, total_rmse_loss, total_mae_loss))

        # save the best model
        if len(self.history['test_RMSE']) == 0 or max(self.history['test_RMSE']) == total_rmse_loss:
            if save_ckpt:
                self.save_checkpoint()
                print("[*] saved best model!")
                open(log_path / self.log_file, 'a').write("[*] saved best model!\n")



    def save_checkpoint(self):
        """Saves model and optimizer to a checkpoint."""
        # save while training
        start_time = time.time()
        checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optim)
        manager = tf.train.CheckpointManager(checkpoint, directory=self.ckpt_dir, max_to_keep=self.checkpoints_to_keep)
        step = self.step.numpy()
        manager.save(checkpoint_number=step)
        logging.info('Saved checkpoint to %s at step %s', self.ckpt_dir, step)
        logging.info('Saving model took %.1f seconds', time.time() - start_time)

    @property
    def step(self):
        """The number of training steps completed."""
        return self.optim.iterations

    def load_checkpoint(self, load_ckpt=''):
        """Restore model and optimizer from a checkpoint if it exists."""
        logging.info('Restoring from checkpoint...')
        start_time = time.time()
        checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optim)
        manager = tf.train.CheckpointManager(checkpoint, directory=self.ckpt_dir, max_to_keep=self.checkpoints_to_keep)
        if load_ckpt == '':
            checkpoint.restore(manager.latest_checkpoint)
        else:
            checkpoint.restore(load_ckpt)
        logging.info('Loaded checkpoint %s', manager.latest_checkpoint)
        logging.info('Loading model took %.1f seconds', time.time() - start_time)









