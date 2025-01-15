import os
from setuptools import setup

if __name__ == "__main__":
    setup(
        name="xnguyen",
        version="0.0.1",
        author="Xuan-Bac Nguyen",
        scripts=[
            "xnguyen/bin/multi_submit.sh",
            "xnguyen/bin/srun_buffer.sh",
            "xnguyen/bin/multi_cluster_vars.sh",
        ],
        author_email="ngxbac.dt@gmail.com",
        description=("Training base code"),
        packages=["xnguyen"],
    )
