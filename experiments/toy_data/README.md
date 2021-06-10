# Toy Experiments

Here we include the code for our analytical toy experiments. We use a toy mixture model and a toy HMM model for analyzing 
the properties of our McBits coders including:
- convergence: the convergence of **net** bitrate to the entropy as `N` increases
- cleanliness: the gap between **net** bitrate and **ideal** bitrate
- initial bit cost: the gap between **total** bitrate and **net** bitrate

You can play around with the toy examples in [`toy_example.py`](./toy_example.py). We also include the notebooks for making the plots in our paper:
- [`initial_bits_plot_toy_mixture`](./initial_bits_plot_toy_mixture.ipynb): the initial bit cost plot on toy mixture model
- [`cleanliness_convergence_plots_toy_mixture`](./cleanliness_convergence_plots_toy_mixture.ipynb): the cleanliness anc convergence plots on toy mixture model
- [`cleanliness_convergence_plots_toy_hmm`](./cleanliness_convergence_plots_toy_hmm.ipynb): the cleanliness anc convergence plots on toy HMM model 