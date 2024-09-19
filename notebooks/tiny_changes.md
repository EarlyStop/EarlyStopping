# Tiny Changes
- [ ] Add method name to the CG print: "Transformed residual vector is zero. Algorithm terminates at iteration 13.
"
- [ ] Add warning message when oracle iteration is not found
- [ ] Update docu landweber
- [ ] Add a proper code documentation for the simmulation wrapper
- [ ] change strong_empirical_errors to strong_empirical_risk in CG (Laura)
- [ ] Make sure to check if all the quantities for computing oracle quantities already exist before just running into an error on the get balanaced oracle methods


# Current goal:
- Make it easy for the user to explore their dataset with our algorithms. This means that we also add 
some major options to the user side. Careful: do not make too many options.

# Next Steps
- [x] Clean up testing_simulation_wrapper and create a class which includes functions for the examples. Maybe use nicer names for variables.
- [x] Apply on CG and landweber
- [x] Make it possible to pass the response from outside
- [x] Add relative efficiencies to the simmulation wrapper
- [ ] Solve error: es.SimulationData.heat(sample_size=100)
- [ ] Take some plotting parameters to the user side. Add option to save plots.
- [ ] Add stopping index in quantities plots
- [ ] Basic visualization of CG
- [ ] Mit Markus reden, nach diesen Schritten. 
