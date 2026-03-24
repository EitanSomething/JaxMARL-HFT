def modify_ippo_rnn(lines):
    # Find start of epoch loop
    epoch_loop_idx = 0
    for i, line in enumerate(lines):
        if "for epoch in range(bc_epochs):" in line:
            epoch_loop_idx = i
            break

    # Setup the JAX eval function before the epoch loop
    setup_code = """
            # Setup pure JAX evaluation function for actual model slippage
            def _bc_eval_step(eval_runner_state, unused):
                train_states_eval, eval_env_state, last_obs, last_done, h_states, eval_rng = eval_runner_state
                actions = []
                new_h_states = []
                for i_ag, train_state_eval in enumerate(train_states_eval):
                    obs_i = batchify(last_obs[i_ag], config["NUM_ACTORS_PERTYPE"][i_ag])
                    ac_in = (obs_i[jnp.newaxis, :], last_done[i_ag][jnp.newaxis, :])
                    h, pi = train_state_eval.apply_fn(train_state_eval.params, h_states[i_ag], ac_in)
                    action = pi.mode().squeeze(0)
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
                
                step_info = []
                for i_ag in range(len(train_states_eval)):
                    step_info.append(info["agents"][i_ag].get("adj_slippage_bps", jnp.zeros_like(reward[i_ag])))
                
                new_runner_state = (train_states_eval, eval_env_state, obsv, done_batch, new_h_states, eval_rng)
                return new_runner_state, step_info

            @jax.jit
            def _run_model_eval(train_states_current, eval_rng):
                eval_rng, reset_rng_key = jax.random.split(eval_rng)
                reset_rng = jax.random.split(reset_rng_key, config["NUM_ENVS"])
                eval_obsv, eval_env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
                
                eval_hstates = []
                init_dones_agents_eval = []
                for i_ag in range(len(train_states_current)):
                    eval_hstates.append(ScannedRNN.initialize_carry(config["NUM_ACTORS_PERTYPE"][i_ag], config["GRU_HIDDEN_DIM"]))
                    init_dones_agents_eval.append(jnp.zeros((config["NUM_ACTORS_PERTYPE"][i_ag]), dtype=bool))
                    
                eval_runner_state_init = (train_states_current, eval_env_state, eval_obsv, init_dones_agents_eval, eval_hstates, eval_rng)
                _, eval_slippage_traj = jax.lax.scan(_bc_eval_step, eval_runner_state_init, None, config["NUM_STEPS"])
                return eval_slippage_traj

"""
    lines.insert(epoch_loop_idx - 1, setup_code)
    
    return "".join(lines)

with open("/home/eitant/Documents/School/ML-Capstone/JaxMARL-HFT/gymnax_exchange/jaxrl/MARL/ippo_rnn_JAXMARL.py", "r") as f:
    lines = f.readlines()
with open("test_bc_patch_2.py", "w") as f:
    f.write(modify_ippo_rnn(lines))

def add_eval_call(lines):
    # Find evaluation injection point
    eval_call_idx = 0
    for i, line in enumerate(lines):
        if "# Compute validation metrics every epoch" in line:
            eval_call_idx = i
            break
            
    eval_call = """
                    # Evaluate ACTUAL model execution slippage
                    rng, _run_eval_rng = jax.random.split(rng)
                    eval_slippage_traj = _run_model_eval(train_states, _run_eval_rng)
                    
                    true_model_slippage_bps = None
                    if i < len(eval_slippage_traj):
                        true_model_slippage_bps = float(np.mean(np.asarray(eval_slippage_traj[i])))
"""
    lines.insert(eval_call_idx - 1, eval_call)
    
    # replace fake string logging with true model
    for i, line in enumerate(lines):
        if "log_dict[f\"bc/agent_{agent_type_names[i]}/price_slippage_bps\"] = price_slippage_bps" in line:
            lines[i] = '                            if true_model_slippage_bps is not None:\n                                log_dict[f"bc/agent_{agent_type_names[i]}/price_slippage_bps"] = true_model_slippage_bps\n'
        elif "slippage_str = f\" price_slippage={price_slippage_bps:.2f}bps\"" in line:
            lines[i] = '                        if true_model_slippage_bps is not None:\n                            slippage_str = f" true_model_slippage={true_model_slippage_bps:.2f}bps"\n'
    return "".join(lines)

with open("test_bc_patch_2.py", "r") as f:
    lines = f.readlines()
with open("test_bc_patch_2.py", "w") as f:
    f.write(add_eval_call(lines))
