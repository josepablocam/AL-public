# AL
A project to automatically generate supervised learning programs
by learning from previously written programs.

We collected approximately 500 programs, extracted dynamic traces, and
build a model to predict the next API call based on previous `j` calls
and features of the current data state.

We previously provided a docker demo, but have now moved on to a simple
API for use. This is meant for demonstration purposes.

# Build
We recommend you create a virtual environment, for example using
conda.

```
conda create -n al-env python=3.6
```

You can then install the requirements for use of AL.

```
pushd src/core
pip install -r requirements.txt
popd
```

# Using AL

You can import `al` from `src/core` for use.


```
pushd src/core
python
```

```
import al
import sklearn.datasets
X, y = sklearn.datasets.make_classification()

m = al.AL()
m.fit(X, y)
progs = m.get_programs()
print(progs[0].pipeline_code())
```
Note that `.pipeline_code` prints out a pipeline without some boilerplate
(such as splitting `X, y` into training and validation).
