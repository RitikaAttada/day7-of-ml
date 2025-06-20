## 🌱 Day 7 of My Machine Learning Journey

Welcome to Day 7 of my Machine Learning journey! 🚀
Today was all about experimentation, precision, and fine-tuning models for better prediction performance on nonlinear mathematical functions.

### 🧠 What I Did

✅ Explored multiple regression models for complex equations like:

* `Y = cos(X * Z)`
* `Y = 0.5X² + log(|Z| + 1)`
* `Y = 1 / (1 + e^-X) + 1 / (1 + e^-Z)`
* `Y = X * Z + 20`
* `Y = X² + Z²`

✅ Understood the importance of:

* Input and output normalization (especially for `tanh`)
* Range matching between model activations and output data
* Using `np.meshgrid` vs `np.random.uniform`
* Flattening vs direct input stacking with `np.column_stack`

✅ Visualized:

* Loss over epochs 📉
* Predicted vs Actual values using scatter plots

### 🧪 Tools Used

* TensorFlow / Keras
* NumPy
* Matplotlib

### 🔍 Key Learnings

* `tanh` requires data in the \[-1, 1] range — normalize accordingly!
* Even simple-looking equations like `cos(X * Z)` can produce complex patterns.
* More neurons and deeper layers sometimes help with precision, but careful tuning is essential.
* Scatter plots are a powerful diagnostic tool to evaluate model performance.
