def inject_bc_eval(lines):
    insertion_idx = 0
    for i, line in enumerate(lines):
        if "Save checkpoint" in line:
            insertion_idx = i
            break
            
    snippet = """
            # Roll out the trained BC model in the environment to calculate actual slippage
            print("Evaluating trained BC model in the environment to calculate true slippage...")
            
            # Use deterministic mode for evaluation
            def _bc_eval_step(eval_runner_state, unused):
                train_states_eval, eval_env_state, last_obs, last_done, h_states, eval_rng = eval_runner_state
                
                actions = []
                new_h_states = []
                
                for i_ag, train_state_eval in enumerate(train_states_eval):
                    obs_i = last_obs[i_ag]
                    # Expand dimensions for network input
                    obs_i = batchify(obs_i, config["NUM_ACTORS_PERTYPE"][i_ag])
                    ac_in = (
                        obs_i[jnp.newaxis, :],
                        last_done[i_ag][jnp.newaxis, :],
                    )
                    
                    # actor_only=True returns h_state, pi (no value)
                    h, pi = train_state_eval.apply_fn(train_state_eval.params, h_states[i_ag], ac_in)
                    action = pi.mode().squeeze(0)  # Use mode for deterministic eval
                    
                    action_unbatched = unbatchify(action, config["NUM_ENVS"], env.multi_agent_config.number_of_agents_per_type[i_ag])
                    actions.append(action_unbatched.squeeze())
                    new_h_states.append(h)
                    
                eval_rng, _rng = jax.random.split(eval_rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                
                obsv, eval_env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, eval_env_state, actions, env_params)
                
                done_batch = []
                for i_ag in range(len(train_states_eval)):
                    d = batchify(done["agents"][i_ag], config["NUM_ACTORS_PERTYPE"][i_ag])
                    done_batch.append(jnp.squeeze(d))
                
                # Extract slippage
                step_info = []
                for i_ag in range(len(train_states_eval)):
                    step_info.append(info["agents"][i_ag].get("adj_slippage_bps", jnp.zeros_like(reward[i_ag])))

                new_runner_state = (train_states_eval, eval_env_state, obsv, done_batch, new_h_states, eval_rng)
                return new_runner_state, step_info
            
            # Setup initial state for evaluation rollout
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
            eval_obsv, eval_env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
            
            eval_hstates = []
            init_dones_agents_eval = []
            for i_ag in range(len(train_states)):
                eval_hstates.append(ScannedRNN.initialize_carry(config["NUM_ACTORS_PERTYPE"][i_ag], config["GRU_HIDDEN_DIM"]))
                init_dones_agents_eval.append(jnp.zeros((config["NUM_ACTORS_PERTYPE"][i_ag]), dtype=bool))
                
            eval_runner_state = (
                train_states,
                eval_env_state,
                eval_obsv,
                init_dones_agents_eval,
                eval_hstates,
                _rng,
            )
            
            # Run scan 
            print("Jitting evaluation loop...")
            bc_eval_scan = jax.jit(lambda state: jax.lax.scan(_bc_eval_step, state, None, config["NUM_STEPS"]))
            _, eval_slippage_traj = bc_eval_scan(eval_runner_state)
            
            # Parse and log metrics
            for i_ag in range(len(train_states)):
                agent_slip = np.asarray(eval_slippage_traj[i_ag])
                actual_slippage = float(np.mean(agent_slip))
                print(f"Agent {i_ag} True Model Mean Slippage: {actual_slippage:.2f} bps")
                if config["WANDB_MODE"] != "disabled" and run is not None:
                    wandb.log({
                        f"bc/agent_{agent_type_names[i_ag]}/true_model_price_slippage_bps": actual_slippage,
                        f"bc/agent_{agent_type_names[i_ag]}/true_model_price_slippage_std": float(np.std(agent_slip))
                    }, commit=False)
"""
    lines.insert(insertion_idx, snippet)
    return "".join(lines)

with open("/home/eitant/Documents/School/ML-Capstone/JaxMARL-HFT/gymnax_exchange/jaxrl/MARL/ippo_rnn_JAXMARL.py", "r") as f:
    lines = f.readlines()

new_content = inject_bc_eval(lines)
with open("/home/eitant/Documents/School/ML-Capstone/JaxMARL-HFT/test_bc_patch.py", "w") as f:
    f.write(new_content)

print("Created test patch")
