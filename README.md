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

# Docker container
Some people have had issues using the install suggested above,
particularly they encounter issues relating to `xgboost`. To address this,
users can instead build a docker container for AL. We refer the
user to https://www.docker.com/ for information on how to install docker.

Once docker is installed, the container can be built by running

```
docker build . -t al-container

```

And it can be launched by running

```
docker run -it -rm al-container
```

which launches a (temporary) container with `bash` as an entry point.
You can then launch the `python` interpreter and import AL following
the steps in the next subsection.

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
