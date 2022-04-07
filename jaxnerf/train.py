# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Training script for Nerf."""

import functools
import gc
import time
from absl import app
from absl import flags
import flax
from flax.metrics import tensorboard
from flax.training import checkpoints
import jax
from jax import config
from jax import random
import jax.numpy as jnp
import numpy as np

from jaxnerf.nerf import datasets
from jaxnerf.nerf import models
from jaxnerf.nerf import utils

FLAGS = flags.FLAGS                                                             #util에서 정의된 flags.FLAGS값들을 모두 가져옴.

utils.define_flags()                                                            #FLAGS값 정의.
config.parse_flags_with_absl()                                                  #FLAGS 값들을 parsing해서 읽어올 수 있게 함.


def train_step(model, rng, state, batch, lr):
  """One optimization step.

  Args:
    model: The linen model.                                                     linen:from flax import linen as nn. flax의 nn이다.
    rng: jnp.ndarray, random number generator.                                  난수 생성기.
    state: utils.TrainState, state of the model/optimizer.                      Optimizer에 대한 Class.
    batch: dict, a mini-batch of data for training.                             데이터들의 미니배치
    lr: float, real-time learning rate.

  Returns:
    new_state: utils.TrainState, new training state.
    stats: list. [(loss, psnr), (loss_coarse, psnr_coarse)].
    rng: jnp.ndarray, updated random number generator.
  """
  rng, key_0, key_1 = random.split(rng, 3)
  
  #loss_fn 시작
  def loss_fn(variables):                                                         #loss, psnr 계산 함수.
    rays = batch["rays"]                                                          #ray 여러개를 가져옴. 배치로 처리함.
    ret = model.apply(variables, key_0, key_1, rays, FLAGS.randomized)            #ray와 변수를 MLP에 집어넣음.
    if len(ret) not in (1, 2):                                                    #MLP 결과의 길이는 1 아니면 2여야함.(fine이거나 coarse,fine이거나)
      raise ValueError(
          "ret should contain either 1 set of output (coarse only), or 2 sets"
          "of output (coarse as ret[0] and fine as ret[1]).")
    # The main prediction is always at the end of the ret list.
    rgb, unused_disp, unused_acc = ret[-1]                                        #모델 결과가 rgb, ?, ?
    loss = ((rgb - batch["pixels"][Ellipsis, :3])**2).mean()                      #실제 데이터의 rgb값과 비교.
    psnr = utils.compute_psnr(loss) #loss to psnr
    if len(ret) > 1:                                                              #MLP 결과의 길이가 2면 coarse한 부분과 fine한 부분의 결과가 동시에 나온다는 뜻. 
      # If there are both coarse and fine predictions, we compute the loss for    
      # the coarse prediction (ret[0]) as well. 
      rgb_c, unused_disp_c, unused_acc_c = ret[0]                                 #len=2 라면 coarse결과에 대한 loss 계산.
      loss_c = ((rgb_c - batch["pixels"][Ellipsis, :3])**2).mean()
      psnr_c = utils.compute_psnr(loss_c)
    else:
      loss_c = 0.
      psnr_c = 0.

    def tree_sum_fn(fn):                                                          #variables 각각에 fn적용시켜 더하는 함수.
      return jax.tree_util.tree_reduce(                                           #tree_util->functions working with tree-like data structures(nested tuples, lists, and dicts)
          lambda x, y: x + fn(y), variables, initializer=0)                       #nested -> (1,2,(3,4),5) 이런 것들. 여기서 leaf의 개수는 5개.
                                                                                  #variables에 대해 lambda x, y: x + fn(y) 를 차례대로 적용시킴.

    weight_l2 = (                                                                 #원소들의 제곱의 합을 원소의 개수로 나눈다.
        tree_sum_fn(lambda z: jnp.sum(z**2)) /                                    #variables 제곱의 합.
        tree_sum_fn(lambda z: jnp.prod(jnp.array(z.shape))))                      #variables 원소들의 개수. z.shape:(2,6) jnp.array(2,6):[2,6] jnp.prod([2,6]):12.
 
    stats = utils.Stats(                                                          #loss, psnr... 가지고 있는 class.
        loss=loss, psnr=psnr, loss_c=loss_c, psnr_c=psnr_c, weight_l2=weight_l2)  
    return loss + loss_c + FLAGS.weight_decay_mult * weight_l2, stats             #loss_fn->loss와 stats을 return. weight_decay_mult = 0(default).
    #loss_fn 끝
    
  (_, stats), grad = (
      jax.value_and_grad(loss_fn, has_aux=True)(state.optimizer.target))          #value_and_grad: (함수값,auxilary data:stats), 도함수값 return.
  grad = jax.lax.pmean(grad, axis_name="batch")                                   #axis_name을 따라 값들의 평균을 냄.
  stats = jax.lax.pmean(stats, axis_name="batch")                                 

  # Clip the gradient by value.
  if FLAGS.grad_max_val > 0:                                                      #The gradient clipping value(default=0).
    clip_fn = lambda z: jnp.clip(z, -FLAGS.grad_max_val, FLAGS.grad_max_val)      #z값을 grad_max_val와 -grad_max_val 사이로 가둠.
    grad = jax.tree_util.tree_map(clip_fn, grad)                                  #grad의 원소에 clip_fn적용.

  # Clip the (possibly value-clipped) gradient by norm.
  if FLAGS.grad_max_norm > 0:                                                     #The gradient clipping magnitude(default=0).
    grad_norm = jnp.sqrt(
        jax.tree_util.tree_reduce(
            lambda x, y: x + jnp.sum(y**2), grad, initializer=0))                 #grad_norm=(grad 제곱의 합)^0.5
    mult = jnp.minimum(1, FLAGS.grad_max_norm / (1e-7 + grad_norm))               
    grad = jax.tree_util.tree_map(lambda z: mult * z, grad)                       #grad의 원소에 mult 곱해줌.

  new_optimizer = state.optimizer.apply_gradient(grad, learning_rate=lr)          #optimizer(Adam) 업데이트.
  new_state = state.replace(optimizer=new_optimizer)
  return new_state, stats, rng
  #train_step 끝.

def main(unused_argv):
  rng = random.PRNGKey(20200823)                                                   #난수 생성기 key
  # Shift the numpy random seed by host_id() to shuffle data loaded by different
  # hosts.
  np.random.seed(20201473 + jax.host_id())                                         #data shuffle

  if FLAGS.config is not None:
    utils.update_flags(FLAGS)
  if FLAGS.batch_size % jax.device_count() != 0:
    raise ValueError("Batch size must be divisible by the number of devices.")
  if FLAGS.train_dir is None:
    raise ValueError("train_dir must be set. None set now.")
  if FLAGS.data_dir is None:
    raise ValueError("data_dir must be set. None set now.")
  dataset = datasets.get_dataset("train", FLAGS)                                    #datasets.py 에서 get_dataset 함수 호출>Blender class 불러옴.
  test_dataset = datasets.get_dataset("test", FLAGS)                                #datasets.py 에서 get_dataset 함수 호출>Blender class 불러옴.
  
  rng, key = random.split(rng)                                                      #난수 생성기 key를 두개로 쪼갬
  model, variables = models.get_model(key, dataset.peek(), FLAGS)                   #(NerfModel, init_variables) return.
  optimizer = flax.optim.Adam(FLAGS.lr_init).create(variables)                      #optimizer 오브젝트 생성. OptimizerDef.create()->optimizer 오브젝트 생성. Adam이 OptimizerDef상속받음.
  state = utils.TrainState(optimizer=optimizer)                                     #state에 Adam optimizer 담음.
  del optimizer, variables

  learning_rate_fn = functools.partial(                                             #utils.learning_rate_decay함수의 인자만 바꾼(partial) 새로운 함수 생성.
      utils.learning_rate_decay,                                                    #learning_rate_decay 나중에 확인!!
      lr_init=FLAGS.lr_init,
      lr_final=FLAGS.lr_final,
      max_steps=FLAGS.max_steps,
      lr_delay_steps=FLAGS.lr_delay_steps,
      lr_delay_mult=FLAGS.lr_delay_mult)

  train_pstep = jax.pmap(
      functools.partial(train_step, model),
      axis_name="batch",
      in_axes=(0, 0, 0, None),
      donate_argnums=(2,))

  def render_fn(variables, key_0, key_1, rays):   
    return jax.lax.all_gather(
        model.apply(variables, key_0, key_1, rays, FLAGS.randomized),
        axis_name="batch")

  render_pfn = jax.pmap(                                                              #render_fn을 pmap시킴. gpu에서 빠르게 rendering가능.
      render_fn,
      in_axes=(None, None, None, 0),  # Only distribute the data input.
      donate_argnums=(3,),
      axis_name="batch",
  )

  # Compiling to the CPU because it's faster and more accurate.
  ssim_fn = jax.jit(
      functools.partial(utils.compute_ssim, max_val=1.), backend="cpu")

  if not utils.isdir(FLAGS.train_dir):
    utils.makedirs(FLAGS.train_dir)
  state = checkpoints.restore_checkpoint(FLAGS.train_dir, state)
  # Resume training a the step of the last checkpoint.
  init_step = state.optimizer.state.step + 1
  state = flax.jax_utils.replicate(state)

  if jax.host_id() == 0:
    summary_writer = tensorboard.SummaryWriter(FLAGS.train_dir)

  # Prefetch_buffer_size = 3 x batch_size
  pdataset = flax.jax_utils.prefetch_to_device(dataset, 3)
  n_local_devices = jax.local_device_count()
  rng = rng + jax.host_id()  # Make random seed separate across hosts.
  keys = random.split(rng, n_local_devices)  # For pmapping RNG keys.
  gc.disable()  # Disable automatic garbage collection for efficiency.
  stats_trace = []
  reset_timer = True
  for step, batch in zip(range(init_step, FLAGS.max_steps + 1), pdataset):
    if reset_timer:
      t_loop_start = time.time()
      reset_timer = False
    lr = learning_rate_fn(step)
    state, stats, keys = train_pstep(keys, state, batch, lr)
    if jax.host_id() == 0:
      stats_trace.append(stats)
    if step % FLAGS.gc_every == 0:
      gc.collect()

    # Log training summaries. This is put behind a host_id check because in
    # multi-host evaluation, all hosts need to run inference even though we
    # only use host 0 to record results.
    if jax.host_id() == 0:
      if step % FLAGS.print_every == 0:
        summary_writer.scalar("train_loss", stats.loss[0], step)
        summary_writer.scalar("train_psnr", stats.psnr[0], step)
        summary_writer.scalar("train_loss_coarse", stats.loss_c[0], step)
        summary_writer.scalar("train_psnr_coarse", stats.psnr_c[0], step)
        summary_writer.scalar("weight_l2", stats.weight_l2[0], step)
        avg_loss = np.mean(np.concatenate([s.loss for s in stats_trace]))
        avg_psnr = np.mean(np.concatenate([s.psnr for s in stats_trace]))
        stats_trace = []
        summary_writer.scalar("train_avg_loss", avg_loss, step)
        summary_writer.scalar("train_avg_psnr", avg_psnr, step)
        summary_writer.scalar("learning_rate", lr, step)
        steps_per_sec = FLAGS.print_every / (time.time() - t_loop_start)
        reset_timer = True
        rays_per_sec = FLAGS.batch_size * steps_per_sec
        summary_writer.scalar("train_steps_per_sec", steps_per_sec, step)
        summary_writer.scalar("train_rays_per_sec", rays_per_sec, step)
        precision = int(np.ceil(np.log10(FLAGS.max_steps))) + 1
        print(("{:" + "{:d}".format(precision) + "d}").format(step) +
              f"/{FLAGS.max_steps:d}: " + f"i_loss={stats.loss[0]:0.4f}, " +
              f"avg_loss={avg_loss:0.4f}, " +
              f"weight_l2={stats.weight_l2[0]:0.2e}, " + f"lr={lr:0.2e}, " +
              f"{rays_per_sec:0.0f} rays/sec")
      if step % FLAGS.save_every == 0:
        state_to_save = jax.device_get(jax.tree_map(lambda x: x[0], state))
        checkpoints.save_checkpoint(
            FLAGS.train_dir, state_to_save, int(step), keep=100)

    # Test-set evaluation.
    if FLAGS.render_every > 0 and step % FLAGS.render_every == 0:
      # We reuse the same random number generator from the optimization step
      # here on purpose so that the visualization matches what happened in
      # training.
      t_eval_start = time.time()
      eval_variables = jax.device_get(jax.tree_map(lambda x: x[0],
                                                   state)).optimizer.target
      test_case = next(test_dataset)
      pred_color, pred_disp, pred_acc = utils.render_image(
          functools.partial(render_pfn, eval_variables),
          test_case["rays"],
          keys[0],
          FLAGS.dataset == "llff",
          chunk=FLAGS.chunk)

      # Log eval summaries on host 0.
      if jax.host_id() == 0:
        psnr = utils.compute_psnr(
            ((pred_color - test_case["pixels"])**2).mean())
        ssim = ssim_fn(pred_color, test_case["pixels"])
        eval_time = time.time() - t_eval_start
        num_rays = jnp.prod(jnp.array(test_case["rays"].directions.shape[:-1]))
        rays_per_sec = num_rays / eval_time
        summary_writer.scalar("test_rays_per_sec", rays_per_sec, step)
        print(f"Eval {step}: {eval_time:0.3f}s., {rays_per_sec:0.0f} rays/sec")
        summary_writer.scalar("test_psnr", psnr, step)
        summary_writer.scalar("test_ssim", ssim, step)
        summary_writer.image("test_pred_color", pred_color, step)
        summary_writer.image("test_pred_disp", pred_disp, step)
        summary_writer.image("test_pred_acc", pred_acc, step)
        summary_writer.image("test_target", test_case["pixels"], step)

  if FLAGS.max_steps % FLAGS.save_every != 0:
    state = jax.device_get(jax.tree_map(lambda x: x[0], state))
    checkpoints.save_checkpoint(
        FLAGS.train_dir, state, int(FLAGS.max_steps), keep=100)


if __name__ == "__main__":
  app.run(main)
