import checkpointed_core

__all__ = ['graph_pipeline']


def graph_pipeline(name: str,
                   steps: list[tuple[str, type[checkpointed_core.PipelineStep]]],
                   connections: list[tuple[str, str, str]],
                   inputs: list[str],
                   outputs: dict[str, str]) -> tuple[checkpointed_core.Pipeline, list]:
    pipeline = checkpointed_core.Pipeline(name)
    handles = []
    handle_mapping = {}
    for name, factory in steps:
        if name in inputs:
            handle = pipeline.add_source(factory,
                                         name=name,
                                         is_sink=name in outputs,
                                         filename=outputs[name] if name in outputs else None)
        elif name in outputs:
            handle = pipeline.add_sink(factory, name=name, filename=outputs[name])
        else:
            handle = pipeline.add_step(factory, name=name)
        handles.append(handle)
        handle_mapping[name] = handle
    for source, target, label in connections:
        pipeline.connect(handle_mapping[source], handle_mapping[target], label)
    return pipeline, handles
