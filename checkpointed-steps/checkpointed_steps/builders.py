import checkpointed

__all__ = ['linear_pipeline']


def linear_pipeline(name: str,
                    *steps: tuple[str, type[checkpointed.PipelineStep]],
                    outputs: dict[str, str]) -> tuple[checkpointed.Pipeline, list]:
    pipeline = checkpointed.Pipeline(name)
    if len(steps) == 0:
        raise ValueError("Linear pipeline must have at least one step")
    handles = []
    source, *remainder = steps
    previous = pipeline.add_source(
        source[1],
        is_sink=source[0] in outputs,
        name=source[0],
        filename=outputs[source[0]] if source[0] in outputs else None,
    )
    handles.append(previous)
    for name, factory in remainder:
        if name in outputs:
            current = pipeline.add_sink(factory, name=name, filename=outputs[name])
        else:
            current = pipeline.add_step(factory, name=name)
        pipeline.connect(previous, current)
        handles.append(current)
        previous, current = current, None
    return pipeline, handles
