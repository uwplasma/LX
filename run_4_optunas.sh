export XLA_PYTHON_CLIENT_PREALLOCATE=false
# export JAX_NUM_THREADS=2
# export OMP_NUM_THREADS=1
# export XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
for i in 1 2; do
  python _hyper_opt.py -c input.toml -n 500 \
    --storage sqlite:///lx_optuna.db \
    --study-name LX-PINN-HPO \
    --n-jobs 4 --keep-trials best &
done
wait