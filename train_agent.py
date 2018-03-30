import sys, pdb, time, random, os, datetime, csv, theano, copy, pickle
import cv2
import numpy as np
from random import randrange
# from ale_python_interface import ALEInterface
import gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import OrderedDict
import pickle as pkl
import theano.tensor as T
import scipy, scipy.misc
from neural_net import OptionCritic_Network
from exp_replay_mujoco import DataSet
from plot_learning import plot

sys.setrecursionlimit(50000)


def load_params(model_path):
  mydir = "/".join(model_path.split("/")[:-1])
  model_params = pkl.load(open(os.path.join(mydir, 'model_params.pkl'), 'rb'))
  return model_params

def create_dir(p):
  try:
    os.makedirs(p)
  except OSError as e:
    if e.errno != 17:
      raise # This was not a "directory exist" error..

def filecreation(model_params, folder_name=None):
  tempdir = os.path.join(os.getcwd(), "models")
  create_dir(tempdir)
  folder_name = folder_name if folder_name is not None else datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
  mydir = os.path.join(tempdir, folder_name)
  create_dir(mydir)
  pkl.dump(model_params, open(os.path.join(mydir, 'model_params.pkl'), "wb"))
  return mydir

class Trainer(object):
  def create_results_file(self):
    self.prog_file = os.path.join(self.mydir, 'training_progress.csv')
    data_file = open(self.prog_file, 'w')
    data_file.write('epoch,mean_score,mean_q_val\n')
    data_file.close()

    self.term_prob_file = os.path.join(self.mydir, 'term_prob.csv')
    data_file = open(self.term_prob_file, 'w')
    data_file.write('epoch,termination_prob\n')
    data_file.close()

  def update_results(self, epoch, ave_reward, ave_q):
    # if it isn't, then we are testing and watching a game.
    # no need to update a file.
    if self.params.nn_file is None:
      fd = open(self.prog_file,'a')
      fd.write('%d,%f,%f\n' % (epoch, ave_reward, ave_q))
      fd.close()
      plot(self.mydir)

  def update_term_probs(self, epoch, term_probs):
    if self.params.nn_file is None:
      fd = open(self.term_prob_file,'a')
      term_probs = term_probs if type(term_probs) is list else [term_probs]
      for term_prob in term_probs:
        fd.write('%d,%f\n' % (epoch, term_prob))
      fd.close()

  def test_dnn(self):
    #chooses which convnet to use based on cudnn availability
    self.params.USE_DNN_TYPE = False
    if theano.config.device.startswith("gpu"):
      self.params.USE_DNN_TYPE=theano.sandbox.cuda.dnn.dnn_available()
    if self.params.USE_DNN_TYPE:
      print("USING CUDNN")
    else:
      print("WARNING: NOT USING CUDNN. TRAINING WILL BE SLOWER.")
    #self.params.USE_DNN_TYPE=False

  def __init__(self, model_params, env_name, folder_name):
    self.init_time = time.time()
    # nn_file only present when watching test
    if model_params.nn_file is None:
      self.mydir = filecreation(model_params, folder_name)
      self.create_results_file()

    self.params = model_params
    self.env = gym.make(env_name)
    self.print_option_stats = model_params.testing
    self.term_ratio = 0
    self.test_dnn()
    self.rng = np.random.RandomState(1234)
    self.noop_action = 0
    self.frame_count = 0.
    self.best_reward = -100000.
    # self.max_frames_per_game = 18000 #TODO: Maybe change this??
    self.num_actions = self.env.action_space.n
    self.obs_shape = self.env.observation_space.shape[0]
    self.legal_actions = range(self.num_actions)
    print("NUM ACTIONS --->", len(self.legal_actions))
    self.action_counter = [{j:0 for j in self.legal_actions} for i in range(self.params.num_options)]

  def _init_ep(self):
    x = self.env.reset()
    return np.asarray([x], dtype=np.float32)

  def act(self, action, testing=False):
    next_obs, reward, done, _ = self.env.step(self.legal_actions[action])
    return np.asarray(next_obs, dtype=np.float32), np.asarray(reward, dtype=np.float32), done

  def get_observation(self):
    x = self.env.get_observation()
    return np.asarray(x, dtype=np.float32)

  def save_model(self, total_reward, skip_best=False):
    if total_reward >= self.best_reward and not skip_best:
      self.best_reward = total_reward
      pkl.dump(self.model.save_params(), open(os.path.join(self.mydir, 'best_model.pkl'), "wb"), protocol=pkl.HIGHEST_PROTOCOL)
    pkl.dump(self.model.save_params(), open(os.path.join(self.mydir, 'last_model.pkl'), "wb"), protocol=pkl.HIGHEST_PROTOCOL)
    print("Saved model")

  def run_training_episode(self):
    raise NotImplementedError

  def get_learning_params(self):
    d = {}
    if self.params.update_rule == "rmsprop":
      d["lr"] = self.params.learning_rate
      d["eps"] = self.params.rms_epsilon
      d["rho"] = self.params.rms_decay
    elif self.params.update_rule == "adam":
      d["lr"] = self.params.learning_rate
    return d

  def get_epsilon(self):
    #linear descent from 1 to 0.1 starting at the replay_start_time
    replay_start_time = max([self.frame_count-self.params.replay_start_size, 0]) 
    epsilon = self.params.epsilon_start
    epsilon -= (self.params.epsilon_start - self.params.epsilon_min)*\
      (min(replay_start_time, self.params.epsilon_decay)/self.params.epsilon_decay)
    return epsilon

  def get_mean_q_val(self, batch=1000):
    imgs = self.exp_replay.random_batch(batch, random_selection=True)
    return np.mean(np.max(self.model.get_q_vals(imgs[0]),axis=1))

  def run_testing(self, epoch):
    total_reward = 0
    num_games = 0
    original_frame_count = self.frame_count 
    rem = self.params.steps_per_test
    while(self.frame_count - original_frame_count < self.params.steps_per_test):
      reward, fps = self.run_training_episode(testing=True)
      print(("TESTING: %d fps,\t" % fps), end=' ')
      self.env.reset()
      print("%d points,\t" % reward, end=' ')
      total_reward += reward
      num_games += 1
    self.frame_count = original_frame_count
    mean_reward = round(float(total_reward)/num_games, 2)
    print("AVERAGE_SCORE:", mean_reward)
    #TODO: Need to do this logging properly
    if type(self) is Q_Learning:
      mean_q = self.get_mean_q_val() if self.params.nn_file is None else 1
    else:
      mean_q = 1
    self.update_results(epoch+1, mean_reward, mean_q)

  def train(self):
    cumulative_reward = 0
    counter = 0
    for i in range(self.params.epochs):
      start_frames = self.frame_count
      frames_rem = self.params.steps_per_epoch #Frames that are remaining
      self.term_probs = []
      while self.frame_count-start_frames < self.params.steps_per_epoch:
        total_reward, fps = self.run_training_episode()
        cumulative_reward += total_reward
        frames_rem = self.params.steps_per_epoch-(self.frame_count-start_frames)
        print(("ep %d,\t") % (counter+1), end=' ')
        print(("%d fps,\t" % fps), end=' ')
        self.env.reset()
        print(('%d points,\t' % total_reward), end=' ')
        print(('%.1f avg,\t' % (float(cumulative_reward)/(counter+1))), end=' ')
        print("%d rem," % frames_rem, 'eps: %.4f' % self.get_epsilon(), end=' ')
        # print("ETA: %d:%02d" % (max(0, frames_rem/60/fps*4), ((frames_rem/fps*4)%60) if frames_rem > 0 else 0), end=' ')
        # print("term ratio %.2f" % (100*self.term_ratio))
        counter += 1

      if self.params.nn_file is None:
        self.save_model(total_reward)
      self.update_term_probs(i, self.term_probs)

      self.run_testing(i)

class DQN_Trainer(Trainer):
  def __init__(self, **kwargs):
    super(DQN_Trainer, self).__init__(**kwargs)

  def run_training_episode(self, testing=False):
    def get_new_frame(new_frame, x):
      return np.asarray([new_frame], dtype=np.float32)

    start_time = time.time()

    total_reward = 0
    data_set = self.test_replay if testing else self.exp_replay
    start_frame_count = self.frame_count
    x = self._init_ep()
    s = self.model.get_state([x])
    done = False
    current_option = 0
    current_action = 0
    new_option = self.model.predict_move(s)[0]
    termination = True
    episode_counter = 0
    termination_counter = 0
    since_last_term = 1
    while not done:
      self.frame_count += 1
      episode_counter += 1
      epsilon = self.get_epsilon() if not testing else self.params.optimal_eps
      if termination:
        if self.print_option_stats:
          print("terminated -------", since_last_term, end=' ')
        termination_counter += 1
        since_last_term = 1
        current_option = np.random.randint(self.params.num_options) if np.random.rand() < epsilon else new_option
        #current_option = self.get_option(epsilon, s)
      else:
        if self.print_option_stats:
          print("keep going", end=' ')
        since_last_term += 1
      current_action = self.model.get_action(s, [current_option])[0]
      #print current_option, current_action
      if self.print_option_stats:
        print(current_option, end=' ')# current_action
        #print [round(i, 2) for i in self.model.get_action_dist(s, [current_option])[0]]
        if True:
          self.action_counter[current_option][self.legal_actions[current_action]] += 1
          data_table = []
          option_count = []
          for ii, aa in enumerate(self.action_counter):
            s3 = sum([aa[a] for a in aa])
            if s3 < 1:
              continue
            print(ii, aa, s3)
            option_count.append(s3)
            print([str(float(aa[a])/s3)[:5] for a in aa])
            data_table.append([float(aa[a])/s3 for a in aa])
            print()

          #ttt = self.model.get_action_dist(s3, [current_option])
          #print ttt, np.sum(-ttt*np.log(ttt))
          print()

      new_frame, reward, done = self.act(current_action, testing=testing)
      data_set.add_sample(x, current_option, reward, done) 

      old_s = copy.deepcopy(s)
      x = get_new_frame(new_frame, x)
      s = self.model.get_state([x])
      term_out = self.model.predict_termination(s, [current_option])
      termination, new_option = term_out[0][0], term_out[1][0]
      if self.frame_count < self.params.replay_start_size and not testing:
        termination = 1
      total_reward += reward
      if self.frame_count > self.params.replay_start_size and not testing:
        self.learn_actor(old_s,
                         np.array([x]),
                         [current_option],
                         [current_action],
                         [reward],
                         [done])
        if self.frame_count % self.params.update_frequency == 0:
          self.learn_critic()
        if self.frame_count % self.params.freeze_interval == 0:
          if self.params.freeze_interval > 999: 
            print("updated_params")
          self.model.update_target_params()

    #print self.last_dist
    self.term_ratio = float(termination_counter)/float(episode_counter)
    if not testing:
      self.term_probs.append(self.term_ratio)
    if self.print_option_stats:
      print("---->", self.term_ratio)
      #self.print_table(data_table, option_count)
    fps = round((self.frame_count - start_frame_count)/(time.time()-start_time), 2)
    # fps = self.ale.getEpisodeFrameNumber()/(time.time()-start_time)
    return total_reward, fps

  def print_table(self, conf_arr, d1):
    pickle.dump(np.array(conf_arr), open( "/".join(self.params.nn_file.split("/")[:-1])+"/confu_data.pkl", "wb" ) )
    self.plot_table(np.array(conf_arr), d1)
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    ax.tick_params(axis='both', which='both',length=0)
    res = ax.imshow(np.array(norm_conf).T, cmap=plt.cm.jet, 
                    interpolation='nearest', vmax=1, vmin=0)

    width, height = conf_arr.shape

    for x in range(width):
        for y in range(height):
            ax.annotate(str(int(conf_arr[x][y]*100)), xy=(x, y), 
                        horizontalalignment='center',
                        verticalalignment='center', size=9)

    cb = fig.colorbar(res)
    plt.xticks(list(range(width)), list(range(1,1+width)))
    plt.yticks(list(range(height)), [self.legal_actions[iii] for iii in range(height)])
    plt.savefig("/".join(self.params.nn_file.split("/")[:-1])+"/"+self.params.nn_file.split("/")[-2].replace(".", "_")+'_confu.png', bbox_inches='tight', format='png')
    raise NotImplemented

  def learn_actor(self, s, next_x, o, a, r, term):
    td_errors = self.model.train_conv_net(s, next_x, o, r, term, actions=a, model="actor")
    return td_errors

  def learn_critic(self):
    x, o, r, next_x, term = self.exp_replay.random_batch(self.params.batch_size)
    td_errors = self.model.train_conv_net(x, next_x, o, r, term, model="critic")
    return td_errors

class Q_Learning(DQN_Trainer):
  def __init__(self, **kwargs):
    super(Q_Learning, self).__init__(**kwargs)
    model_network = [{"model_type": "mlp", "out_size": 64, "activation": "relu"},
                     {"model_type": "mlp", "out_size": 64, "activation": "relu"},
                     {"model_type": "mlp", "out_size": 64, "activation": "relu"},
                     {"model_type": "mlp", "out_size": len(self.legal_actions), "activation": "linear"}]

    learning_params = self.get_learning_params()

    #TODO: check clip delta
    #TODO: check freeze_interval
    #TODO: termination_reg
    #TODO: actor lr
    #TODO: Check temp

    self.model = OptionCritic_Network(model_network=model_network,
      learning_method=self.params.update_rule, dnn_type=self.params.USE_DNN_TYPE, clip_delta=self.params.clip_delta,
      input_size=[None,1,self.env.observation_space.shape[0]], batch_size=self.params.batch_size, learning_params=learning_params,
      gamma=self.params.discount, freeze_interval=self.params.freeze_interval,
      termination_reg=self.params.termination_reg, num_options=self.params.num_options,
      actor_lr=self.params.actor_lr, double_q=self.params.double_q, temp=self.params.temp,
      entropy_reg=self.params.entropy_reg, BASELINE=self.params.baseline)

    if self.params.nn_file is not None:
      self.model.load_params(pkl.load(open(self.params.nn_file, 'r')))

    self.exp_replay = DataSet(self.env.observation_space.shape[0], self.rng, max_steps=self.params.replay_memory_size, phi_length=1) #TODO: Need to change this
    self.test_replay = DataSet(self.env.observation_space.shape[0], self.rng, max_steps=1, phi_length=1) #TODO: Need to change this

if __name__ == "__main__":
  pass
