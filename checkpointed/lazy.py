from .step import PipelineStep


class LazyExecutor:
    """The lazy executor is meant to solve the problem where
    executing two steps sequentially would not make sense.

    Examples of this are loading a file, and then
    immediately discarding it in the store/load procedure.

    The LazyExecutor works by deferring execution of the
    previous steps to the step that can actually use
    the result. Instead of the result of a step,
    the lazy executor itself is returned, together
    with all necessary state to later resume execution.

    When the LazyExecutor is passed to a step, the step
    can call either the "execute" of "stream" method
    in order to perform the work done by the executor.

    Because the LazyExecutor bypasses the regular
    processing pipeline, no checkpointing is performed.

    The LazyExecutor uses pickle to store and load
    its internal state. The only guarantees are
    1) A LazyExecutor can be used as a source
    2) LazyExecutors can be chained
    3) Chained LazyExecutors can have un-pickleable _results_
    4) A LazyExecutor cannot be a sink (results will not be saved)
    """

    def __init__(self, factory: type[PipelineStep], inputs, config):
        self._factory = factory
        self._input = inputs
        self._config = config

    async def execute(self):
        await self._factory(self._config).execute(*self._input)
