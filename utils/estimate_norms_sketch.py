    # Estimate norms 
    num_trials = 256
    norms = np.empty(num_trials, dtype=float)

    for t in tqdm(range(num_trials)):
        
        ei = np.zeros((n, 1), dtype=float)
        ei[np.random.randint(n), 0] = 1.0
    
        FX = srft_sketch(X_train, k=k_val_LinMix, n=n, transform=transform)   # shape (k, d)
        Fe1 = srft_sketch(ei,     k=k_val_LinMix, n=n, transform=transform) # shape (k, 1)

        v = FX.T @ Fe1              # shape (d, 1)
        norms[t] = np.linalg.norm(v, ord=2)

    print("mean "   + str(float(np.mean(norms))))
    print("std "    + str(float(np.std(norms, ddof=1))))
    print("q05 "    + str(float(np.quantile(norms, 0.05))))
    print("median " + str(float(np.quantile(norms, 0.50))))
    print("q95 "    + str(float(np.quantile(norms, 0.95))))