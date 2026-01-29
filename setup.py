# /home/d2l/d2l/setup.py
#本包用于d2l学习过程中的个人调试与更新，用于重构d2l/torch.py
#还包含 /home/d2l/d2l/pyproject.toml用于构建本地pkg
from setuptools import setup, find_packages

setup(
    name="myd2l",
    author='Ewiggestrige_D',
    version="0.0.1",
    packages=find_packages(),  # 会自动包含 myd2l/ 包
    description='personal use/devs',
)