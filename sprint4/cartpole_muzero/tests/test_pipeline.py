import tempfile
import unittest
import inspect
from pathlib import Path
from unittest import mock
import numpy as np
import torch
import gymnasium as gym
from PIL import Image

from cartpole_muzero.buffers import LatentReplayBuffer
from cartpole_muzero.environment import make_state_env, StateObservationWrapper
from cartpole_muzero.diagnose import (
    build_argument_parser as build_diagnostic_argument_parser,
    record_episode_gif,
)
from cartpole_muzero.mcts import MCTSNode, ModelBasedMCTS, visit_count_policy
from cartpole_muzero.models import DynamicsModel, RepresentationNetwork, PredictionNetwork, normalize_latent
from cartpole_muzero.train import (
    CHECKPOINT_VERSION,
    POLICY_TARGET_MODE,
    WARMUP_POLICY_MODE,
    bootstrapped_value_targets,
    build_argument_parser,
    companion_checkpoint_path,
    evaluate,
    load_latent_checkpoint,
    make_latent_mcts,
    save_latent_checkpoint,
    save_reward_plot,
    scale_gradient,
    select_training_action,
    temperature_for_episode,
    train_latent_muzero,
    train_latent_step,
    updates_for_episode,
    validate_training_arguments,
)


def episode(length=4, terminal=True, policy_valid=True):
    return [{"observation":np.full((4,),i/10,np.float32),"action":i%2,"reward":float(i+1),
             "policy":np.array([.8,.2],np.float32) if i%2==0 else np.array([.1,.9],np.float32),
             "policy_valid":policy_valid,"value":float(10+i),
             "terminated":terminal and i==length-1,"truncated":False} for i in range(length)]


class ToyDynamics:
    def __init__(self,rewards): self.rewards=rewards; self.calls=0
    def predict(self,state,action): self.calls+=1; return np.asarray([[[state.item()+action+1]]],np.float32),float(self.rewards[action])

class ToyPrediction:
    def __init__(self,values=None,priors=(.5,.5)): self.values=values or {}; self.priors=np.asarray(priors,np.float32)
    def predict(self,state): return self.priors,float(self.values.get(float(np.asarray(state).item()),0.0))


class FakeDiscrete:
    n=2
    def seed(self,_seed): pass
    def sample(self): return 0


class FakeStateEnv:
    action_space=FakeDiscrete()
    def __init__(self): self.steps=0
    def reset(self,seed=None):
        self.steps=0
        return np.zeros(4,np.float32),{}
    def step(self,_action):
        self.steps+=1
        observation=np.full(4,self.steps/10,np.float32)
        return observation,2.0,self.steps>=2,False,{}
    def close(self): pass


class FakeGifEnv(FakeStateEnv):
    def __init__(self):
        super().__init__()
        self.actions=[]
        self.closed=False

    def step(self,action):
        self.actions.append(action)
        observation,reward,terminated,truncated,info=super().step(action)
        observation.fill(self.steps*64/255)
        return observation,reward,terminated,truncated,info

    def render(self):
        frame=np.zeros((24,36,3),np.uint8)
        frame[...,0]=40+self.steps*60
        return frame

    def close(self): self.closed=True


class FakeGifRepresentation:
    def __init__(self): self.observations=[]
    def encode(self,observation):
        self.observations.append(np.asarray(observation).copy())
        return np.asarray([[[np.asarray(observation).mean()]]],np.float32)


class FakeGifPrediction:
    def predict(self,_latent): return np.asarray([.75,.25],np.float32),.5


class FakeGifMCTS:
    def __init__(self): self.noise=[]
    def search(self,_latent,add_exploration_noise=False):
        self.noise.append(add_exploration_noise)
        root=MCTSNode(); root.children={0:MCTSNode(.5),1:MCTSNode(.5)}
        root.children[0].visit_count=1; root.children[1].visit_count=3
        return root


class MuZeroPipelineTest(unittest.TestCase):
    def make_models(self):
        return RepresentationNetwork(8,16), DynamicsModel(8,2,16), PredictionNetwork(8,2,16)

    def test_native_observations_match_cartpole_without_rendering(self):
        env=make_state_env(); reference=gym.make('CartPole-v1')
        try:
            with mock.patch.object(env.unwrapped,'render',side_effect=AssertionError('render called')):
                observation,_=env.reset(seed=101)
                expected,_=reference.reset(seed=101)
                np.testing.assert_array_equal(observation,expected)
                self.assertEqual(observation.shape,(4,))
                self.assertEqual(observation.dtype,np.float32)
                for action in (0,1,1,0):
                    observation,*_=env.step(action)
                    expected,*_=reference.step(action)
                    np.testing.assert_array_equal(observation,expected)
                invalid=observation.copy(); invalid[0]=np.nan
                with self.assertRaisesRegex(ValueError,'four finite'):
                    env.observation(invalid)
        finally:
            env.close(); reference.close()

    def test_environment_rejects_non_four_dimensional_inputs(self):
        env=gym.make('Pendulum-v1')
        try:
            with self.assertRaisesRegex(ValueError,'four-dimensional'):
                StateObservationWrapper(env)
        finally:
            env.close()

    def test_constant_latents_normalize_safely(self):
        state=torch.ones(2,8,requires_grad=True)
        normalized=normalize_latent(state)
        torch.testing.assert_close(normalized,torch.zeros_like(normalized))
        normalized.square().mean().backward()
        self.assertTrue(torch.isfinite(state.grad).all())

    def test_representation_is_vector_bounded_finite_differentiable(self):
        rep,_,_=self.make_models(); x=torch.rand(3,4); y=rep(x)
        self.assertEqual(y.shape,(3,8)); self.assertTrue(torch.isfinite(y).all())
        self.assertGreaterEqual(y.min(),0); self.assertLessEqual(y.max(),1)
        y.square().mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in rep.parameters()))
        np.testing.assert_allclose(rep.encode(x[0].numpy()),y[0].detach().numpy(),atol=1e-6)
        with self.assertRaisesRegex(ValueError,'observation must have shape'):
            rep(torch.zeros(1,5))

    def test_dynamics_shape_and_action_response(self):
        _,dyn,_=self.make_models(); s=torch.rand(2,8)
        a,_=dyn(s,torch.zeros(2,dtype=torch.long)); b,_=dyn(s,torch.ones(2,dtype=torch.long))
        self.assertEqual(a.shape,s.shape); self.assertFalse(torch.allclose(a,b)); self.assertTrue(torch.isfinite(a).all()); self.assertTrue(((a>=0)&(a<=1)).all())

    def test_prediction_shapes_are_logits_and_scalar(self):
        _,_,pred=self.make_models(); logits,value=pred(torch.rand(3,8)); self.assertEqual(logits.shape,(3,2)); self.assertEqual(value.shape,(3,1))
        self.assertFalse(torch.allclose(logits.sum(1),torch.ones(3)))

    def test_fresh_prediction_starts_with_neutral_policy_prior(self):
        _,_,prediction=self.make_models()
        probabilities,_=prediction.predict(np.random.rand(8).astype(np.float32))
        np.testing.assert_allclose(probabilities,[.5,.5],atol=1e-7)

    def test_vector_networks_work_end_to_end_for_different_latent_sizes(self):
        for latent_dim,batch_size in ((8,1),(32,3)):
            representation=RepresentationNetwork(latent_dim,16)
            dynamics=DynamicsModel(latent_dim,2,16)
            prediction=PredictionNetwork(latent_dim,2,16)
            root=representation(torch.rand(batch_size,4))
            logits,value=prediction(root)
            recurrent,reward=dynamics(root,torch.zeros(batch_size,dtype=torch.long))
            self.assertEqual(root.shape,(batch_size,latent_dim))
            self.assertEqual(logits.shape,(batch_size,2))
            self.assertEqual(value.shape,(batch_size,1))
            self.assertEqual(recurrent.shape,root.shape)
            self.assertEqual(reward.shape,(batch_size,1))
            predicted,reward_scalar=dynamics.predict(root[0].detach().numpy(),1)
            self.assertEqual(predicted.shape,(latent_dim,))
            self.assertTrue(np.isfinite(reward_scalar))
            probabilities,_=prediction.predict(predicted)
            self.assertAlmostEqual(float(probabilities.sum()),1.,places=6)

    def test_replay_alignment_root_plus_k(self):
        replay=LatentReplayBuffer(); replay.add_episode(episode()); sample=replay.sample(1,3,2,starts=[0])
        _,actions,policies,values,rewards,pm,vm,rm=sample
        self.assertEqual(actions.tolist(),[[0,1,0]]); self.assertEqual(rewards[:, :, 0].tolist(),[[1.,2.,3.]])
        self.assertEqual(values[:,:,0].tolist(),[[10.,11.,12.,13.]])
        self.assertTrue(torch.all(pm==1)); self.assertTrue(torch.all(vm==1)); self.assertTrue(torch.all(rm==1))

    def test_replay_checkpoint_state_roundtrip(self):
        original=LatentReplayBuffer(capacity=17); original.add_episode(episode(3))
        restored=LatentReplayBuffer(); restored.load_state_dict(original.state_dict())
        self.assertEqual(restored.capacity,17); self.assertEqual(len(restored),3)
        self.assertEqual(restored.episodes[0][2]["reward"],3.0)

    def test_exactly_root_plus_k_predictions(self):
        rep,dyn,pred=self.make_models(); replay=LatentReplayBuffer(); replay.add_episode(episode())
        opt=torch.optim.Adam([*rep.parameters(),*dyn.parameters(),*pred.parameters()])
        head_calls=[]
        hook=pred.policy_hidden.register_forward_hook(lambda *_args:head_calls.append(1))
        result=train_latent_step(rep,dyn,pred,replay,opt,1,3,2,torch.device('cpu'))
        hook.remove()
        self.assertEqual(result['prediction_calls'],4); self.assertEqual(len(head_calls),4)

    def test_weighted_total_uses_unweighted_component_losses(self):
        rep,dyn,pred=self.make_models(); replay=LatentReplayBuffer(); replay.add_episode(episode(1))
        opt=torch.optim.Adam([*rep.parameters(),*dyn.parameters(),*pred.parameters()],lr=0)
        result=train_latent_step(rep,dyn,pred,replay,opt,1,1,2,torch.device('cpu'),
            policy_loss_weight=2,value_loss_weight=3,reward_loss_weight=4)
        expected=2*result['policy_loss']+3*result['value_loss']+4*result['reward_loss']
        self.assertAlmostEqual(result['total_loss'],expected,places=4)

    def test_gradient_scale_and_clip_are_configurable(self):
        value=torch.ones(3,requires_grad=True); scale_gradient(value,.3).sum().backward()
        self.assertTrue(torch.allclose(value.grad,torch.full((3,),.3)))
        rep,dyn,pred=self.make_models(); replay=LatentReplayBuffer(); replay.add_episode(episode(1))
        opt=torch.optim.Adam([*rep.parameters(),*dyn.parameters(),*pred.parameters()],lr=0)
        original_clip=torch.nn.utils.clip_grad_norm_
        with mock.patch('cartpole_muzero.train.torch.nn.utils.clip_grad_norm_',wraps=original_clip) as clip:
            train_latent_step(rep,dyn,pred,replay,opt,1,1,2,torch.device('cpu'),gradient_clip_norm=.75)
        self.assertEqual(clip.call_args.args[1],.75)

    def test_terminal_absorbing_padding(self):
        replay=LatentReplayBuffer(); replay.add_episode(episode(1)); _,a,p,v,r,pm,vm,_=replay.sample(1,3,2,starts=[0])
        self.assertEqual(a.tolist(),[[0,0,0]]); self.assertEqual(r[:,:,0].tolist(),[[1.,0.,0.]])
        self.assertEqual(v[:,:,0].tolist(),[[10.,0.,0.,0.]]); self.assertEqual(pm[:,:,0].tolist(),[[1.,0.,0.,0.]])
        self.assertEqual(vm[:,:,0].tolist(),[[1.,1.,1.,1.]])

    def test_warmup_replay_masks_only_policy_targets(self):
        replay=LatentReplayBuffer(); replay.add_episode(episode(2,policy_valid=False))
        _,_,_,values,rewards,policy_masks,value_masks,reward_masks=replay.sample(
            1,1,2,starts=[0])
        self.assertEqual(policy_masks[:,:,0].tolist(),[[0.,0.]])
        self.assertEqual(value_masks[:,:,0].tolist(),[[1.,1.]])
        self.assertEqual(reward_masks[:,:,0].tolist(),[[1.]])
        self.assertEqual(values[:,:,0].tolist(),[[10.,11.]])
        self.assertEqual(rewards[:,:,0].tolist(),[[1.]])

    def test_all_policy_targets_masked_still_train_value_reward_and_models(self):
        torch.manual_seed(7)
        rep,dyn,pred=self.make_models(); replay=LatentReplayBuffer()
        replay.add_episode(episode(4,policy_valid=False))
        opt=torch.optim.Adam([*rep.parameters(),*dyn.parameters(),*pred.parameters()],lr=0)
        result=train_latent_step(rep,dyn,pred,replay,opt,4,2,2,torch.device('cpu'))
        self.assertEqual(result['policy_loss'],0.)
        for key in ('total_loss','value_loss','reward_loss'):
            self.assertTrue(np.isfinite(result[key])); self.assertGreater(result[key],0.)
        for net in (rep,dyn,pred):
            gradients=[parameter.grad for parameter in net.parameters()
                       if parameter.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
            self.assertGreater(sum(gradient.abs().sum().item() for gradient in gradients),0.)
        policy_gradients=[
            parameter.grad
            for layer in (pred.policy_hidden,pred.policy_head)
            for parameter in layer.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(policy_gradients)
        self.assertEqual(sum(gradient.abs().sum().item() for gradient in policy_gradients),0.)

    def test_no_episode_crossing(self):
        replay=LatentReplayBuffer(); replay.add_episode(episode(1)); other=episode(1); other[0]['reward']=99; replay.add_episode(other)
        for _ in range(10):
            rewards=replay.sample(1,3,2,starts=[0])[4][0,:,0].tolist(); self.assertIn(rewards[0],[1.,99.]); self.assertEqual(rewards[1:],[0.,0.])

    def test_joint_step_gives_all_networks_gradients(self):
        rep,dyn,pred=self.make_models(); replay=LatentReplayBuffer(); replay.add_episode(episode())
        opt=torch.optim.Adam([*rep.parameters(),*dyn.parameters(),*pred.parameters()],lr=1e-3)
        train_latent_step(rep,dyn,pred,replay,opt,4,3,2,torch.device('cpu'))
        for net in (rep,dyn,pred):
            grads=[p.grad for p in net.parameters() if p.grad is not None]; self.assertTrue(grads); self.assertTrue(all(torch.isfinite(g).all() for g in grads)); self.assertGreater(sum(g.abs().sum().item() for g in grads),0)

    def test_mcts_prefers_higher_reward(self):
        m=ModelBasedMCTS(ToyDynamics([0,2]),ToyPrediction(),2,simulations=30,discount=.9); root=m.search(np.zeros((1,1,1),np.float32)); self.assertGreater(root.children[1].visit_count,root.children[0].visit_count)

    def test_mcts_prefers_higher_leaf_value(self):
        values={1.:0.,2.:3.}; m=ModelBasedMCTS(ToyDynamics([0,0]),ToyPrediction(values),2,simulations=3,discount=.9); root=m.search(np.zeros((1,1,1),np.float32)); self.assertGreater(root.children[1].visit_count,root.children[0].visit_count)

    def test_one_edge_expanded_per_simulation(self):
        dyn=ToyDynamics([0,0]); m=ModelBasedMCTS(dyn,ToyPrediction(),2,simulations=7); m.search(np.zeros((1,1,1),np.float32)); self.assertEqual(dyn.calls,7); self.assertEqual(m.expansions_last_search,7)

    def test_noise_only_when_requested(self):
        np.random.seed(1); m=ModelBasedMCTS(ToyDynamics([0,0]),ToyPrediction(priors=(.9,.1)),2,simulations=1)
        clean=m.search(np.zeros((1,1,1),np.float32),False); np.random.seed(1); noisy=m.search(np.zeros((1,1,1),np.float32),True)
        self.assertAlmostEqual(clean.children[0].prior,.9,places=5); self.assertNotAlmostEqual(noisy.children[0].prior,.9,places=5)

    def test_mcts_hyperparameters_are_plumbed(self):
        mcts=make_latent_mcts(ToyDynamics([0,0]),ToyPrediction(),2,simulations=7,
            discount=.91,pb_c_base=123,pb_c_init=2.5,root_dirichlet_alpha=.4,
            root_exploration_fraction=.15)
        self.assertEqual(mcts.simulations,7); self.assertEqual(mcts.discount,.91)
        self.assertEqual(mcts.pb_c_base,123); self.assertEqual(mcts.pb_c_init,2.5)
        self.assertEqual(mcts.root_dirichlet_alpha,.4); self.assertEqual(mcts.root_exploration_fraction,.15)

    def test_fixed_batch_overfits_all_losses(self):
        torch.manual_seed(0); rep,dyn,pred=self.make_models()
        replay=LatentReplayBuffer(); replay.add_episode(episode(2))
        sample=replay.sample
        # Hold the evaluation/training batch fixed instead of comparing different random roots.
        with mock.patch.object(replay,'sample',side_effect=lambda b,k,a,d:sample(b,k,a,d,starts=[0,1])):
            opt=torch.optim.Adam([*rep.parameters(),*dyn.parameters(),*pred.parameters()],lr=5e-3)
            first=train_latent_step(rep,dyn,pred,replay,opt,2,1,2,torch.device('cpu'))
            for _ in range(300):
                last=train_latent_step(rep,dyn,pred,replay,opt,2,1,2,torch.device('cpu'))
        self.assertLess(last['policy_loss'],first['policy_loss']*.8)
        self.assertLess(last['value_loss'],first['value_loss']*.01)
        self.assertLess(last['reward_loss'],first['reward_loss']*.01)

    def test_checkpoint_roundtrip(self):
        rep,dyn,pred=self.make_models()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.pt'; save_latent_checkpoint(path,rep,dyn,pred,{'note':'ok'}); loaded=load_latent_checkpoint(path)
            self.assertEqual(loaded[3]['checkpoint_version'],CHECKPOINT_VERSION); self.assertEqual(loaded[3]['note'],'ok')
            self.assertEqual(loaded[3]['policy_target_mode'],POLICY_TARGET_MODE)
            self.assertEqual(loaded[3]['warmup_policy_mode'],WARMUP_POLICY_MODE)
            self.assertEqual(loaded[3]['latent_dim'],rep.latent_dim)
            self.assertEqual(loaded[3]['observation_mode'],'cartpole_state_4d')
            self.assertEqual(loaded[2].latent_dim,rep.latent_dim)
            for original,restored in zip((rep,dyn,pred),loaded[:3]):
                for name,value in original.state_dict().items():
                    torch.testing.assert_close(value,restored.state_dict()[name],rtol=0,atol=0)
            self.assertEqual(list(Path(d).glob('*.tmp')),[])

    def test_reward_plot_is_created(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'rewards.png'
            result=save_reward_plot({'scores':[1,2,3], 'evaluations':[(3,2.5)]},path)
            self.assertEqual(result,path); self.assertTrue(path.exists()); self.assertGreater(path.stat().st_size,0)

    def test_old_checkpoints_are_rejected_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'old.pt'
            for version in (2,8,11,12,13,14):
                with self.subTest(version=version):
                    torch.save({'checkpoint_version':version},path)
                    with self.assertRaisesRegex(ValueError,rf'version {version}.*4D input.*fresh run'):
                        load_latent_checkpoint(path)

    def test_checkpoint_observation_mode_is_validated(self):
        rep,dyn,pred=self.make_models()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'wrong.pt'; save_latent_checkpoint(path,rep,dyn,pred)
            data=torch.load(path,weights_only=False); data['observation_mode']='screenshots'
            torch.save(data,path)
            with self.assertRaisesRegex(ValueError,'cartpole_state_4d'):
                load_latent_checkpoint(path)

    def test_n_step_bootstrap_and_terminal_zero(self):
        data=episode(4,False); data[2]['value']=7
        self.assertAlmostEqual(bootstrapped_value_targets(data,.5,2)[0],1+.5*2+.25*7)
        data[1]['terminated']=True; self.assertAlmostEqual(bootstrapped_value_targets(data,.5,2)[0],2.)

    def test_reward_scale_applies_to_reward_sum_not_bootstrap_again(self):
        data=episode(4,False); data[2]['value']=7
        target=bootstrapped_value_targets(data,.5,2,reward_scale=.1)[0]
        self.assertAlmostEqual(target,.1+.5*.2+.25*7)

    def test_temperature_schedule_boundaries(self):
        values=[temperature_for_episode(i,1,.5,.25,2,4) for i in range(6)]
        self.assertEqual(values,[1,1,.5,.5,.25,.25])

    def test_training_target_uses_raw_visits_not_behavior_temperature(self):
        root=MCTSNode()
        root.children={0:MCTSNode(),1:MCTSNode()}
        root.children[0].visit_count=40; root.children[1].visit_count=10
        behavior=visit_count_policy(root,.25)
        with mock.patch('cartpole_muzero.mcts.np.random.choice',return_value=0) as choose:
            action,target=select_training_action(root,.25)
        self.assertEqual(action,0)
        np.testing.assert_allclose(choose.call_args.kwargs['p'],behavior,atol=1e-7)
        np.testing.assert_allclose(target,[.8,.2],atol=1e-7)
        self.assertGreater(behavior[0],.99)
        self.assertFalse(np.allclose(target,behavior))

    def test_update_frequency_fixed_and_transition_relative(self):
        self.assertEqual(updates_for_episode(100,7,0,1,50),7)
        self.assertEqual(updates_for_episode(10,7,.25,5,50),5)
        self.assertEqual(updates_for_episode(500,7,.25,5,50),50)

    def test_parser_accepts_stable_example_and_maps_every_argument(self):
        parser=build_argument_parser()
        arguments=vars(parser.parse_args([
            '--env','CartPole-v1','--episodes','800','--max-steps','500',
            '--latent-dim','64','--hidden-dim','128',
            '--simulations','50','--batch-size','64','--buffer-capacity','50000',
            '--warmup-episodes','20','--updates-per-transition','.25',
            '--min-updates-per-episode','1','--max-updates-per-episode','50',
            '--learning-rate','1e-4','--weight-decay','1e-5','--gradient-clip-norm','5',
            '--dynamics-gradient-scale','.5','--discount','.99','--bootstrap-steps','10',
            '--unroll-steps','5','--reward-scale','.1','--policy-loss-weight','1',
            '--value-loss-weight','1','--reward-loss-weight','1','--pb-c-base','19652',
            '--pb-c-init','1.25','--root-dirichlet-alpha','.25',
            '--root-exploration-fraction','.25','--temperature-initial','1',
            '--temperature-middle','.5','--temperature-final','.25',
            '--temperature-initial-episodes','200','--temperature-middle-episodes','500',
            '--evaluation-interval','20','--evaluation-episodes','20',
            '--checkpoint-path','checkpoints/test.pt','--reuse-checkpoint','--seed','3',
        ]))
        function_parameters=set(inspect.signature(train_latent_muzero).parameters)
        self.assertEqual(set(arguments),function_parameters-{'device'})
        self.assertEqual(arguments['env_id'],'CartPole-v1'); self.assertEqual(arguments['reward_scale'],.1)
        self.assertEqual(arguments['updates_per_transition'],.25); self.assertTrue(arguments['reuse_checkpoint'])

    def test_defaults_use_vector_architecture_and_distinct_checkpoint(self):
        arguments=build_argument_parser().parse_args([])
        self.assertEqual(arguments.temperature_initial_episodes,200)
        self.assertEqual(arguments.temperature_middle_episodes,500)
        self.assertEqual(arguments.min_updates_per_episode,1)
        self.assertTrue(arguments.checkpoint_path.endswith('cartpole_state_v15.pt'))

    def test_diagnostic_parser_accepts_gif_options(self):
        arguments=build_diagnostic_argument_parser().parse_args([
            'model.pt','--record-gif','artifacts/run.gif','--gif-mode','raw',
            '--gif-seed','17','--gif-max-steps','23','--gif-fps','12','--gif-width','320',
        ])
        self.assertEqual(arguments.checkpoint,'model.pt')
        self.assertEqual(arguments.record_gif,Path('artifacts/run.gif'))
        self.assertEqual(arguments.gif_mode,'raw'); self.assertEqual(arguments.gif_seed,17)
        self.assertEqual(arguments.gif_max_steps,23); self.assertEqual(arguments.gif_fps,12)
        self.assertEqual(arguments.gif_width,320)

    def test_diagnostic_records_native_inputs_and_noise_free_mcts_gifs(self):
        with tempfile.TemporaryDirectory() as directory:
            for mode,expected_action in (('raw',0),('mcts',1)):
                with self.subTest(mode=mode):
                    env=FakeGifEnv(); mcts=FakeGifMCTS()
                    representation=FakeGifRepresentation(); path=Path(directory)/f'{mode}.gif'
                    with mock.patch('cartpole_muzero.diagnose.make_state_env',return_value=env):
                        result=record_episode_gif(path,'Fake-v0',representation,FakeGifPrediction(),mcts,
                            mode=mode,seed=7,max_steps=4,fps=10,output_width=360)
                    self.assertEqual(env.actions,[expected_action]*2); self.assertTrue(env.closed)
                    self.assertEqual(result['score'],4.); self.assertEqual(result['steps'],2)
                    self.assertEqual(result['observation_mode'],'cartpole_state_4d')
                    self.assertEqual(len(representation.observations),2)
                    self.assertEqual(representation.observations[0].shape,(4,))
                    np.testing.assert_allclose(representation.observations[0],0)
                    np.testing.assert_allclose(representation.observations[1],64/255)
                    self.assertEqual(mcts.noise,[False,False] if mode=='mcts' else [])
                    with Image.open(path) as image:
                        self.assertEqual(image.format,'GIF'); self.assertEqual(image.n_frames,3)
                        self.assertGreater(image.width,360)

    def test_evaluation_repeats_seeds_without_changing_actor_rng(self):
        rep,dyn,pred=self.make_models()
        mcts=make_latent_mcts(dyn,pred,2,simulations=3)
        np.random.seed(123); before=np.random.get_state()
        first=evaluate('CartPole-v1',rep,pred,mcts,2,15,[20100,20101])
        after=np.random.get_state()
        np.testing.assert_equal(before,after)
        second=evaluate('CartPole-v1',rep,pred,mcts,2,15,[20100,20101])
        self.assertEqual(first,second)

    def test_validation_rejects_invalid_search_and_temperature_values(self):
        arguments=vars(build_argument_parser().parse_args([]))
        arguments['simulations']=0
        with self.assertRaisesRegex(ValueError,'simulations must be positive'):
            validate_training_arguments(**arguments)
        arguments=vars(build_argument_parser().parse_args([]))
        arguments['root_exploration_fraction']=1.1
        with self.assertRaisesRegex(ValueError,'root_exploration_fraction'):
            validate_training_arguments(**arguments)
        arguments=vars(build_argument_parser().parse_args([]))
        arguments['temperature_initial_episodes']=3; arguments['temperature_middle_episodes']=2
        with self.assertRaisesRegex(ValueError,'temperature episodes'):
            validate_training_arguments(**arguments)

    def test_training_metadata_scaling_updates_and_resume_optimizer_override(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint=Path(directory)/'run.pt'
            common=dict(env_id='Fake-v0',episodes=1,max_steps=2,latent_dim=8,hidden_dim=16,simulations=2,batch_size=2,buffer_capacity=20,
                warmup_episodes=0,updates_per_episode=1,updates_per_transition=.5,
                min_updates_per_episode=2,max_updates_per_episode=2,discount=.9,
                bootstrap_steps=2,unroll_steps=1,reward_scale=.1,evaluation_interval=1,
                evaluation_episodes=1,checkpoint_path=checkpoint,seed=4,device=torch.device('cpu'))
            with mock.patch('cartpole_muzero.train.make_state_env',side_effect=lambda *_args,**_kwargs:FakeStateEnv()), \
                 mock.patch('cartpole_muzero.train.save_reward_plot',return_value=Path(directory)/'plot.png'):
                train_latent_muzero(**common,learning_rate=1e-3,weight_decay=1e-4)
                train_latent_muzero(**common,learning_rate=2e-4,weight_decay=2e-5,reuse_checkpoint=True)
            data=torch.load(companion_checkpoint_path(checkpoint,'latest'),weights_only=False)
            self.assertEqual(data['completed_episodes'],2); self.assertEqual(len(data['raw_episode_scores']),2)
            self.assertEqual(data['replay_state']['size'],4); self.assertAlmostEqual(data['replay_state']['episodes'][0][0]['reward'],.2)
            self.assertEqual(data['update_mode'],'transition-relative(rate=0.5, min=2, max=2)')
            self.assertEqual(data['training_config']['gradient_clip_norm'],5.0)
            group=data['optimizer_state_dict']['param_groups'][0]
            self.assertEqual(group['lr'],2e-4); self.assertEqual(group['weight_decay'],2e-5)

    def test_warmup_episode_updates_learner_and_marks_policy_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint=Path(directory)/'warmup.pt'
            configuration=dict(
                env_id='Fake-v0',episodes=2,max_steps=2,latent_dim=8,hidden_dim=16,simulations=2,batch_size=2,buffer_capacity=20,
                warmup_episodes=1,updates_per_episode=1,updates_per_transition=0,
                min_updates_per_episode=1,max_updates_per_episode=1,discount=.9,
                bootstrap_steps=2,unroll_steps=1,reward_scale=.1,
                evaluation_interval=2,evaluation_episodes=1,
                checkpoint_path=checkpoint,seed=9,device=torch.device('cpu'),
            )
            with mock.patch(
                'cartpole_muzero.train.make_state_env',
                side_effect=lambda *_args,**_kwargs:FakeStateEnv(),
            ), mock.patch(
                'cartpole_muzero.train.save_reward_plot',
                return_value=Path(directory)/'plot.png',
            ):
                train_latent_muzero(**configuration)
            data=torch.load(
                companion_checkpoint_path(checkpoint,'latest'),
                weights_only=False,
            )
            replay_episodes=data['replay_state']['episodes']
            self.assertEqual(len(replay_episodes),2)
            self.assertTrue(all(not item['policy_valid'] for item in replay_episodes[0]))
            self.assertTrue(all(item['policy_valid'] for item in replay_episodes[1]))
            self.assertEqual(len(data['history']['losses']),2)
            self.assertEqual(data['history']['losses'][0]['policy_loss'],0.)
            self.assertGreater(data['history']['losses'][0]['value_loss'],0.)
            self.assertGreater(data['history']['losses'][0]['reward_loss'],0.)

if __name__=='__main__': unittest.main()
