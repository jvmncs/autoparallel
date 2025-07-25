import time

import modal

app = modal.App("test-fc")


@app.function()
def inner(x: str):
    return x


@app.function()
def tester(x: str, y: str):
    a = inner.spawn(x)
    b = inner.spawn(y)
    return [a.get() + "+", b.get() + "-"]


if __name__ == "__main__":
    with modal.enable_output():
        with app.run():
            fc = tester.remote("hello", "hi")
            time.sleep(5)
            call_graph = fc.get_call_graph()

            def print_call_graph(graph, indent=""):
                print(f"{indent}Call Graph:")
                if isinstance(graph, list):
                    graph = graph[0] if graph else {}
                for child in getattr(graph, "children", []):
                    print(f"{indent}  - {getattr(child, 'function_name', 'Unknown')}")
                    print_call_graph(child, indent + "    ")

            print(call_graph)
            print_call_graph(call_graph)
