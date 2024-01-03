pip install -e checkpointed-core
pip install -e checkpointed-steps
python -m build checkpointed-core
python -m build checkpointed-steps
python -m build checkpointed
twine upload checkpointed-core\dist\*
twine upload checkpointed-steps\dist\*
twine upload checkpointed\dist\*
pip uninstall checkpointed-steps
pip uninstall checkpointed-core