"""Spawn-safe worker. Existing orchestrator code is imported only for real runs."""
import os
import time
import traceback


def run_request(connection, request, gated, backend, log_path):
    # Keep dependencies' console output out of the coordinator's CSV/console.
    with open(log_path, 'w', encoding='utf-8', buffering=1) as log:
        import contextlib
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            phase = 'startup'
            def emit(event, **fields):
                connection.send(dict(event=event, time=time.monotonic(),
                                     cpu_time=time.process_time(), pid=os.getpid(), **fields))
            try:
                emit('start')
                phase = 'prepare'
                if backend == 'simulate':
                    emit('stage_start', stage='prepare')
                    time.sleep(request.get('prepare_seconds', .05))
                    if request.get('fail_stage') == 'prepare':
                        raise RuntimeError('injected preparation failure')
                    emit('stage_end', stage='prepare')
                    emit('ready')
                    if gated and connection.recv() != 'infer':
                        raise RuntimeError('invalid inference gate message')
                    phase = 'inference'
                    emit('inference_start')
                    time.sleep(request.get('inference_seconds', .1))
                    if request.get('fail_stage') == 'inference':
                        raise RuntimeError('injected inference failure')
                    emit('done', success=True, response_chars=9)
                else:
                    import orchestrator as workload
                    from langgraph.graph import StateGraph

                    def stage(name, function):
                        def wrapped(state):
                            emit('stage_start', stage=name)
                            try:
                                return function(state)
                            finally:
                                emit('stage_end', stage=name)
                        return wrapped

                    def inference(state):
                        nonlocal phase
                        emit('ready')
                        if gated and connection.recv() != 'infer':
                            raise RuntimeError('invalid inference gate message')
                        phase = 'inference'
                        emit('inference_start')
                        result = workload.final_answer(state)
                        if not str(result.get('final_response', '')).strip():
                            raise RuntimeError('empty final response')
                        return result

                    graph = StateGraph(workload.GraphState)
                    graph.add_node('web_search', stage('web_search', workload.web_search))
                    graph.add_node('fetch_url', stage('fetch_url', workload.fetch_url))
                    graph.add_node('summarize', stage('summarize', workload.summarize))
                    graph.add_node('final_answer', inference)
                    graph.set_entry_point('web_search')
                    graph.add_edge('web_search', 'fetch_url')
                    graph.add_edge('fetch_url', 'summarize')
                    graph.add_edge('summarize', 'final_answer')
                    graph.set_finish_point('final_answer')
                    state = dict(query=request['query'], job_id=request['request_id'],
                                 skip_web_search=request.get('skip_web_search', True),
                                 urls=[], page_texts=[], summaries=[], final_response='')
                    result = graph.compile().invoke(state, config={'max_concurrency': 1})
                    emit('done', success=True,
                         response_chars=len(str(result['final_response'])))
            except BaseException as exc:
                traceback.print_exc()
                try:
                    emit('done', success=False, phase=phase,
                         error=f'{type(exc).__name__}: {exc}')
                except (BrokenPipeError, EOFError, OSError):
                    pass
            finally:
                connection.close()
